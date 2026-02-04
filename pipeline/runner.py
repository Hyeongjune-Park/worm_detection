# pipeline/runner.py — Human-in-the-Loop Multi-Sensor Tracking Pipeline
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from io_utils.video_reader import VideoReader
from io_utils.video_writer import OverlayWriter
from io_utils.artifacts import ArtifactWriter, EventWriter
from io_utils.overlay import draw_overlay, DebugInfo
from io_utils.debug_logger import DebugLogger, FrameDebugRecord, save_debug_snapshot
from qa.shape_analyzer import analyze_mask, ShapeStats
from config_toggles import FeatureToggles

from preprocess.preprocess import Preprocessor
from tracking.track import Track, TrackState
from tracking.kalman import KalmanFilter2D
from tracking.state_machine import TrackStateMachine
from roi.roi_manager import RoiManager, RoiWindow
from sensors.template_sensor import TemplateSensor
from sensors.klt_sensor import KltSensor
from sensors.sam2_sensor import build_sam2_sensor
from sensors.head_estimator import HeadEstimator
from qa.fusion import fuse


def _load_seeds(seeds_path: str) -> List[Dict[str, Any]]:
    with open(seeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("seeds", [])


def _get_arena_rect(cfg: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    arena = cfg.get("arena", {})
    mr = arena.get("manual_rect", {})
    if arena.get("enabled", False) and mr.get("enabled", False):
        return (int(mr["x0"]), int(mr["y0"]), int(mr["x1"]), int(mr["y1"]))
    return None


def _create_tracks_from_seeds(
    seeds: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Track]:
    """seeds.yaml에서 트랙 생성."""
    kcfg = cfg.get("tracking", {}).get("kalman", {})
    q = float(kcfg.get("process_noise_q", 2.0))
    r = float(kcfg.get("default_measurement_noise_r", 25.0))

    tracks = []
    for seed in seeds:
        tid = int(seed["track_id"])
        bbox = seed["bbox"]  # [x0, y0, x1, y1]
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        kf = KalmanFilter2D(cx, cy, q=q, r=r)
        tr = Track(id=tid, kf=kf)
        tr.bbox = (int(x0), int(y0), int(x1), int(y1))
        tr.seed_bbox_size = (int(x1 - x0), int(y1 - y0))  # 시드 박스 크기 저장
        tr.last_center = (cx, cy)
        tr.state = TrackState.ACTIVE
        tracks.append(tr)

    return tracks


def run_pipeline(
    cfg: Dict[str, Any],
    input_path: str,
    output_dir: str,
    seeds_path: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 토글 초기화 ---
    toggles = FeatureToggles.from_config(cfg)
    print(toggles.summary())

    # --- 로드 ---
    seeds = _load_seeds(seeds_path)
    if not seeds:
        print("[ERROR] seeds.yaml is empty. Run select_seeds.py first.")
        return

    arena_rect = _get_arena_rect(cfg)

    reader = VideoReader(input_path, fps_fallback=cfg["video"]["fps_fallback"])
    fps = reader.fps
    W, H = reader.width, reader.height
    frame_size = (W, H)

    # --- 모듈 초기화 (toggles 전달) ---
    pre = Preprocessor(cfg)
    roi_mgr = RoiManager(cfg)
    tpl_sensor = TemplateSensor(cfg)
    klt_sensor = KltSensor(cfg)
    sam2_sensor = build_sam2_sensor(cfg, toggles=toggles)
    head_est = HeadEstimator(cfg)
    state_machine = TrackStateMachine(cfg, toggles=toggles)

    # sensor noise values
    sam2_r = float(cfg.get("sensors", {}).get("sam2", {}).get("measurement_noise_r", 10.0))
    tpl_r = float(cfg.get("sensors", {}).get("template", {}).get("measurement_noise_r", 25.0))
    klt_r = float(cfg.get("sensors", {}).get("klt", {}).get("measurement_noise_r", 25.0))

    # --- 트랙 생성 ---
    tracks = _create_tracks_from_seeds(seeds, cfg)
    print(f"[INIT] {len(tracks)} tracks from seeds")

    # --- 출력 ---
    artifacts = ArtifactWriter(
        out_dir,
        save_csv=cfg["project"].get("save_csv", True),
        fps=fps,
    )
    event_writer: Optional[EventWriter] = None
    if cfg["project"].get("save_events", True):
        event_writer = EventWriter(out_dir / "events.jsonl")

    overlay_writer: Optional[OverlayWriter] = None
    if cfg["project"].get("save_overlay_video", False):
        fourcc = cfg["video"].get("overlay_fourcc", "mp4v")
        overlay_writer = OverlayWriter(
            str(out_dir / "overlay.mp4"), fps=fps, frame_size=frame_size, fourcc=fourcc
        )

    # DebugLogger 초기화 + 토글 기록
    debug_logger = DebugLogger(str(out_dir), fps=fps)
    debug_logger.record_toggles(toggles)
    # 트랙별 이전 ShapeStats 저장
    prev_shape_stats: Dict[int, ShapeStats] = {}

    # --- 첫 프레임: 센서 초기화 ---
    first_frame = True

    # --- 메인 루프 ---
    frame_idx = 0
    for frame_bgr in reader:
        frame_idx += 1
        proc = pre.apply(frame_bgr)
        frame_gray = proc["gray"]

        # 첫 프레임: 센서 초기화
        if first_frame:
            for tr in tracks:
                cx, cy = tr.center()
                roi = roi_mgr.make_roi(cx, cy, frame_size)
                crop_bgr = roi_mgr.crop(frame_bgr, roi)
                crop_gray = roi_mgr.crop(frame_gray, roi)
                tpl_sensor.initialize(tr, roi, crop_bgr, crop_gray)
                klt_sensor.initialize(tr, roi, crop_bgr, crop_gray)
                sam2_sensor.initialize(tr, roi, crop_bgr, crop_gray)
            first_frame = False

        # --- Per-track 처리 ---
        debug_infos: Dict[int, DebugInfo] = {}

        for tr in tracks:
            if tr.state in (TrackState.EXITED, TrackState.NEEDS_RESEED):
                continue

            prev_state = tr.state

            # a) Kalman predict
            tr.kf.predict(dt=1.0)
            tr.update_speed(fps)
            pred_center = tr.kf.get_position()

            # b) ROI 확장 계산
            is_active = (tr.state == TrackState.ACTIVE)

            # [P2] 상태 기반 ROI 확장
            if toggles.state_roi_expansion and tr.state in (TrackState.UNCERTAIN, TrackState.OCCLUDED):
                state_expansion = 1.25
            else:
                state_expansion = 1.0

            # [P1] 속도 기반 ROI 확장 (ACTIVE에서만)
            speed_expansion = 1.0
            if toggles.speed_roi_expansion and is_active and tr.last_center is not None:
                dx = pred_center[0] - tr.last_center[0]
                dy = pred_center[1] - tr.last_center[1]
                dist = (dx * dx + dy * dy) ** 0.5
                roi_base = roi_mgr.base_size
                r = dist / roi_base
                if r > 0.40:
                    speed_expansion = 2.0
                elif r > 0.25:
                    speed_expansion = 1.5

            # [P3] 쿨다운 히스테리시스
            if toggles.expansion_cooldown:
                if is_active:
                    desired_expansion = max(state_expansion, speed_expansion)
                    if desired_expansion > 1.0:
                        tr.expansion_cooldown = 5
                    elif tr.expansion_cooldown > 0:
                        tr.expansion_cooldown -= 1
                        desired_expansion = max(desired_expansion, 1.5)
                else:
                    tr.expansion_cooldown = 0
                    desired_expansion = state_expansion
            else:
                desired_expansion = max(state_expansion, speed_expansion)
                tr.expansion_cooldown = 0

            expansion = desired_expansion
            roi = roi_mgr.make_roi(pred_center[0], pred_center[1], frame_size,
                                   expansion_factor=expansion)
            crop_bgr = roi_mgr.crop(frame_bgr, roi)
            crop_gray = roi_mgr.crop(frame_gray, roi)

            if crop_bgr.size == 0 or crop_gray.size == 0:
                continue

            # c) 센서 측정
            # [P5] REACQUIRE 시 box prompt 확장
            effective_box_expansion = expansion if toggles.reacquire_box_expansion else 1.0
            sam2_result = sam2_sensor.measure(tr, roi, crop_bgr, crop_gray, frame_idx,
                                              box_expansion=effective_box_expansion)
            tpl_result = tpl_sensor.measure(tr, roi, crop_bgr, crop_gray, frame_idx)
            klt_result = klt_sensor.measure(tr, roi, crop_bgr, crop_gray, frame_idx)

            # d) Fusion + QA
            is_merged = (tr.state == TrackState.MERGED)
            roi_size_val = float(roi.x1 - roi.x0)
            # [S4] 이전 프레임 면적 (캐싱 토글에 따라)
            prev_area = sam2_sensor._prev_areas.get(tr.id) if toggles.sam2_mask_caching else None
            fusion = fuse(
                pred_center=pred_center,
                sam2=sam2_result,
                tpl=tpl_result,
                klt=klt_result,
                cfg=cfg,
                roi_size=roi_size_val,
                sam2_r=sam2_r,
                tpl_r=tpl_r,
                klt_r=klt_r,
                is_merged=is_merged,
                prev_area=prev_area,
                track_state=tr.state,
                toggles=toggles,
            )

            # [S4] SAM2 QA 통과 시 prev_mask 캐시 확정
            if toggles.sam2_mask_caching and fusion.sensor_used == "SAM2":
                sam2_sensor.cache_good_result(tr.id)

            # e) Kalman update
            if fusion.do_kalman_update:
                tr.kf.set_measurement_noise(fusion.measurement_r)
                tr.kf.update(fusion.center[0], fusion.center[1])
            else:
                # [K1] predict-only 속도 감쇠
                if toggles.velocity_decay_on_predict:
                    tr.kf.decay_velocity(0.5)

            # f) Track 필드 갱신
            tr.last_center = fusion.center
            tr.sensor_used = fusion.sensor_used
            tr.quality_score = fusion.quality_score
            tr.quality_history.append(fusion.quality_score)
            tr.last_seen_frame = frame_idx

            # [P4] bbox 갱신
            if toggles.sam2_bbox_sync:
                # SAM2 마스크 품질 ok면 sensor_used와 무관하게 동기화
                sam2_bbox_ok = False
                if sam2_result is not None and sam2_result.bbox_roi is not None:
                    border_ok = sam2_result.border_touch < 0.3
                    area_check_available = (prev_area is not None and sam2_result.area is not None and prev_area > 0)
                    if area_check_available:
                        area_ratio = sam2_result.area / prev_area
                        area_ok = 0.3 < area_ratio < 3.0
                    else:
                        area_ok = False
                    has_consensus_bbox = fusion.debug.get("has_consensus", 0.0) > 0.5
                    if is_active:
                        sam2_bbox_ok = border_ok and area_ok and has_consensus_bbox
                    else:
                        sam2_bbox_ok = border_ok and area_ok
            else:
                # BASE: sensor_used == "SAM2"일 때만 bbox 동기화
                sam2_bbox_ok = (
                    sam2_result is not None
                    and sam2_result.bbox_roi is not None
                    and fusion.sensor_used == "SAM2"
                )

            if sam2_bbox_ok:
                full_bbox = roi_mgr.to_full_coords_bbox(sam2_result.bbox_roi, roi)
                tr.bbox = full_bbox
                tr.area = sam2_result.area
            elif fusion.do_kalman_update and fusion.center is not None:
                fcx, fcy = fusion.center
                if tr.bbox is not None:
                    bw = tr.bbox[2] - tr.bbox[0]
                    bh = tr.bbox[3] - tr.bbox[1]
                else:
                    bw, bh = 60, 60
                half_w, half_h = bw // 2, bh // 2
                tr.bbox = (int(fcx - half_w), int(fcy - half_h),
                           int(fcx + half_w), int(fcy + half_h))

            # g) Head 추정
            velocity = tr.kf.get_velocity()
            tr.last_head = head_est.estimate(
                sam2_result, velocity,
                crop_gray=crop_gray,
                roi_origin=(roi.x0, roi.y0),
                fused_center=fusion.center,
            )

            # h) 상태머신
            new_state = state_machine.update(tr, fusion, frame_idx, arena_rect, frame_size)
            tr.state = new_state

            # [S2] Template update (품질 게이팅)
            if fusion.sensor_used == "SAM2":
                if not toggles.template_update_gating or fusion.quality_score >= 0.7:
                    tpl_sensor.update_template(tr, roi, crop_gray, fusion.quality_score)

            # [S3] KLT reinit on divergence
            if toggles.klt_divergence_reinit:
                if fusion.sensor_used == "SAM2" and klt_result is not None:
                    from math import sqrt
                    klt_sam2_dist = sqrt((klt_result.center[0] - fusion.center[0])**2 +
                                         (klt_result.center[1] - fusion.center[1])**2)
                    if klt_sam2_dist > roi_size_val * 0.15:
                        klt_sensor.initialize(tr, roi, crop_bgr, crop_gray)

            # j) Immobility check
            state_machine.check_immobility(tr, fps)

            # k) Event logging
            if event_writer and new_state != prev_state:
                event_writer.log(frame_idx, tr.id, "STATE_CHANGE", {
                    "from": prev_state.value, "to": new_state.value,
                    "sensor": fusion.sensor_used,
                    "quality": fusion.quality_score,
                })

            # l) 디버그 정보 수집
            debug_infos[tr.id] = DebugInfo(
                track_id=tr.id,
                roi=(roi.x0, roi.y0, roi.x1, roi.y1),
                pred_center=pred_center,
                sam2_center=sam2_result.center if sam2_result else None,
                tpl_center=tpl_result.center if tpl_result else None,
                klt_center=klt_result.center if klt_result else None,
                sam2_mask=sam2_result.mask if sam2_result else None,
            )

            # m) 상세 디버그 로깅
            shape_stats = None
            if sam2_result is not None and sam2_result.mask is not None:
                prev_ss = prev_shape_stats.get(tr.id)
                roi_shape = (roi.y1 - roi.y0, roi.x1 - roi.x0)
                shape_stats = analyze_mask(sam2_result.mask, prev_ss, roi_shape)
                prev_shape_stats[tr.id] = shape_stats

            def _dist(a, b):
                if a is None or b is None:
                    return -1.0
                return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

            sam2_c = sam2_result.center if sam2_result else None
            tpl_c = tpl_result.center if tpl_result else None
            klt_c = klt_result.center if klt_result else None

            area_ratio_prev = 1.0
            if prev_area is not None and sam2_result is not None and sam2_result.area is not None and prev_area > 0:
                area_ratio_prev = sam2_result.area / prev_area

            debug_rec = FrameDebugRecord(
                frame_idx=frame_idx,
                t_sec=frame_idx / fps,
                track_id=tr.id,
                state=tr.state.value,
                sensor_used=fusion.sensor_used,
                do_kalman_update=fusion.do_kalman_update,
                measurement_r=fusion.measurement_r,
                quality_score=fusion.quality_score,
                pred_center=pred_center,
                sam2_center=sam2_c,
                tpl_center=tpl_c,
                klt_center=klt_c,
                chosen_center=fusion.center,
                d_pred_sam2=_dist(pred_center, sam2_c),
                d_pred_tpl=_dist(pred_center, tpl_c),
                d_pred_klt=_dist(pred_center, klt_c),
                d_sam2_tpl=_dist(sam2_c, tpl_c),
                d_sam2_klt=_dist(sam2_c, klt_c),
                roi_xyxy=(roi.x0, roi.y0, roi.x1, roi.y1),
                roi_scale=expansion,
                bbox_xyxy=tr.bbox,
                bbox_update="UPDATE" if sam2_bbox_ok else "HOLD",
                shape_stats=shape_stats,
                area_ratio_prev=area_ratio_prev,
                fusion_debug=fusion.debug,
            )
            debug_logger.add_record(debug_rec)

            if debug_rec.reason_flags:
                save_debug_snapshot(frame_bgr, debug_rec, str(out_dir))

        # --- Merge detection ---
        merge_events = state_machine.check_merges(tracks)
        if event_writer:
            for mevt in merge_events:
                event_writer.log(frame_idx, mevt["track_id"], mevt["type"], mevt)

        # --- 출력 ---
        artifacts.write_frame(frame_idx, tracks, frame_size=frame_size, fps=fps)

        if overlay_writer is not None:
            overlay = draw_overlay(frame_bgr.copy(), tracks, frame_idx, arena_rect, debug_infos)
            overlay_writer.write(overlay)

    # --- 정리 ---
    reader.close()
    if overlay_writer is not None:
        overlay_writer.close()
    artifacts.close()
    if event_writer is not None:
        event_writer.close()

    # DebugLogger 저장
    debug_logger.save_all()

    print(f"[OK] done. {frame_idx} frames processed. output -> {out_dir.resolve()}")
