# insect_tracking — Design & Implementation Spec (Human-in-the-Loop, Center/Head Tracking)

> 목표: 실험실 환경(풀잎/물결/저대비/노이즈/겹침/가림/탈출/사망 등)에서도 **최대 10개 애벌레의 위치(중심점) 궤적을 정확히 기록**하는 추적 파이프라인을 구축한다.  
> 핵심 전략: **사람 seed(초기 + 중대 오류 시 재지정)** + **트랙별 ROI 기반** + **다중 센서 교차검증(QA)** + **상태머신**.  
> 전제: 라벨링 기반 추가 학습은 불가하지만, **SAM2처럼 추가 학습 없이 사용 가능한 모델은 사용 가능**.

---

## 1. Goals

### 1.1 Primary Goal
- 각 애벌레(트랙)별 **center (x, y)**를 프레임별/시간별로 기록한다.

### 1.2 Secondary Goal
- 가능할 때만 **head (x, y)**를 추정한다.
- 불확실한 경우 head는 억지로 찍지 않고 `None/NaN` 처리한다.

### 1.3 Non-goal
- 애벌레의 **정밀 테두리(segmentation mask)의 품질 자체**가 최종 목표가 아니다.
- mask는 **좌표(centroid/axis/endpoints) 측정 수단**이며, 결과로 저장은 가능하나 핵심 출력은 좌표다.

---

## 2. Constraints & Operating Assumptions

### 2.1 Constraints
- **라벨링 데이터로 추가 학습 불가.**
- 사전학습 모델(예: SAM2) 사용 가능.
- 영상은 길고 GPU 메모리가 제한될 수 있음 → 필요시 청크 처리.

### 2.2 Human-in-the-loop is allowed (and preferred)
- **초기 seed**: 영상 시작 시 사용자(사람)가 개체별 박스를 지정.
- **중간 reseed**: 드리프트/겹침/가림 등 중대 오류가 감지되면 해당 트랙만 재지정.

> “완전 자동 탐지”를 목표로 하지 않는다.  
> 실험 계측 정확도 최우선 → 최소 비용 사용자 개입을 설계에 포함한다.

---

## 3. High-level Pipeline (Modules & Responsibilities)

### 3.1 Arena Selection (per video)
- 사용자가 영상당 1회 **실험 영역(arena) 사각형**을 지정한다.
- arena 밖은 모든 단계에서 무시한다. (자막/테두리/접시 밖 노이즈 차단)
- 산출물: `arena.yaml`

### 3.2 Seed Selection (per video)
- 사용자가 **최대 10개** 애벌레에 대해 초기 박스를 지정한다.
- 겹침(두 마리 겹쳐 있음)도 사용자가 **각각 박스를 따로 지정** 가능해야 한다.
- 산출물: `seeds.yaml` (track_id별 초기 bbox)

### 3.3 Tracking Loop (per frame, per track)
- 각 트랙에 대해:
  1) Kalman 예측
  2) ROI 설정(트랙 주변 작은 패치)
  3) 다중 센서로 위치 관측치 측정(SAM2 / Template / KLT)
  4) QA(교차검증)로 관측치 신뢰도 평가
  5) 융합(fusion) 후 center/head 업데이트
  6) 상태머신 업데이트(ACTIVE/UNCERTAIN/OCCLUDED/…)
  7) 이벤트 로그 기록

### 3.4 Failure Handling & Reseed
- QA 실패가 누적되면 트랙을 `NEEDS_RESEED`.
- 사용자 재seed 후 즉시 복구.

### 3.5 Output
- `tracks.csv` 또는 `tracks.parquet`: 프레임별 좌표 + 센서/QA/상태.
- `overlay.mp4`: ROI/center/head/track_id/state 시각화.
- `events.jsonl`: merge/exit/reseed 등 이벤트 로그.

---

## 4. Track Data Model

트랙은 “가설(hypothesis)”이며, 관측이 끊겨도 즉시 삭제하지 않는다.

### 4.1 Track Fields (recommended)
- `track_id: int`
- `state: enum`:
  - `ACTIVE`, `UNCERTAIN`, `OCCLUDED`, `MERGED`, `EXITED`, `DEAD_CANDIDATE`, `NEEDS_RESEED`
- Kalman:
  - `kf_state`: `[x, y, vx, vy]` and covariance `P`
- Last known:
  - `last_center: (x, y)`
  - `last_head: (x, y) or None`
  - `last_bbox: (x0, y0, x1, y1)` (optional debug)
- ROI:
  - `roi: (x0, y0, x1, y1)`
- Sensors memory:
  - `template`: patch/feature representation (for template matching)
  - `klt_points`: tracked points (optional)
- Quality:
  - `quality_history: deque[float]`
  - `miss_count: int`
  - `reseed_count: int`
  - `last_good_frame_idx: int`

---

## 5. ROI Design (Critical)

### 5.1 Definition
- ROI = 트랙의 “있을 법한 위치” 주변의 작은 **사각형 crop**.
- ROI는 매 프레임 업데이트되며, 연산은 ROI 내부에서만 수행한다.

### 5.2 ROI Center Update Rule
- 기본 ROI 중심은 **Kalman 예측(center_pred)**.
- 센서 관측치가 QA를 통과할 때만 Kalman 업데이트로 예측을 보정한다.
- **센서 1회 실패로 ROI 중심이 급격히 이동하면 드리프트 폭발** → 금지.

### 5.3 ROI Size Policy (safe defaults)
- ROI 크기는 고정 또는 제한적 가변(상한/하한 필수).
- Recommended initial:
  - `roi_size_base = 384`
  - `roi_size_min = 256`
  - `roi_size_max = 640`
- ROI는 항상 arena 안으로 clamp.

### 5.4 Do NOT: Infinite ROI expansion
- 관측 실패 시 ROI를 계속 키우면 배경을 포함해 오탐 폭발 → 금지.
- 실패 시 `UNCERTAIN/OCCLUDED → NEEDS_RESEED`로 운영.

---

## 6. Sensors (No-label Learning)

> 원칙: SAM2 출력도 “정답”이 아니다.  
> **여러 센서의 좌표 관측치를 교차검증**하고, 불일치하면 보수적으로 처리한다.

### 6.1 Sensor A — SAM2 in ROI (segmentation -> coordinates)
**Purpose**: mask 품질이 아니라 **좌표 관측치**를 얻기 위한 센서.

- Input:
  - ROI crop
  - hint: seed bbox/points or previous step output (implementation choice)
- Output:
  - `mask` (ROI coordinates)
  - `center_sam2`: centroid of mask
  - `axis`: PCA major axis direction
  - `endpoints`: two candidates along axis (or skeleton endpoints)
  - optional: `bbox_sam2` for debug
- Notes:
  - 풀잎을 물고 끌면 mask/bbox가 커지는 것은 “정상”일 수 있음.
  - 따라서 area/bbox 변화만으로 rollback하지 않는다.
  - **좌표 일관성(예측/템플릿/KLT)**이 핵심.

### 6.2 Sensor B — Template Matching in ROI (recommended)
**Purpose**: SAM2 drift/occlusion에 대비한 안정 앵커.

- Seed 시 template 저장:
  - 추천: gray + edge(Sobel magnitude/Canny) 기반 patch
- Each frame:
  - ROI 내 NCC(normalized cross correlation)로 `center_tpl` 산출
- Template update:
  - 드리프트 방지 위해 **QA가 매우 좋은 경우에만** 천천히 업데이트.

### 6.3 Sensor C — KLT Optical Flow in ROI (optional)
**Purpose**: template이 약하거나 질감 변화가 있을 때 보강.

- Initialize:
  - `goodFeaturesToTrack` in ROI
- Track:
  - `calcOpticalFlowPyrLK`
- Robustness:
  - RANSAC 등으로 outlier 제거 후 `center_klt` 추정
- Reinit:
  - 유효점이 충분히 줄면 재초기화

### 6.4 Global motion/blob
- 평상시 주 엔진에서 제외.
- 센서가 모두 불신인 상황에서만 “복구 후보” 힌트로 제한적으로 사용.

---

## 7. QA / Fusion (Core Reliability)

### 7.1 Past failure to avoid
- `bbox too big/small => rollback to previous bbox` 방식은 실패.
- 이유: 풀잎을 물고 움직일 때 bbox가 커지는 게 정상인데, 계속 과거로 되돌아가 anchor/청크 연결이 붕괴.

### 7.2 QA Philosophy
- “마스크 면적”은 보조 지표.
- 최우선은 **좌표 관측치의 일관성**:
  - SAM2 vs Kalman prediction
  - SAM2 vs Template
  - SAM2 vs KLT
- ROI 경계 접촉(잘림/드리프트)을 강한 위험 신호로 본다.

### 7.3 Recommended metrics
- `center_pred`: Kalman predicted center
- `center_sam2`: SAM2 measured center
- `center_tpl`: template matching center (if available)
- `center_klt`: KLT center (if available)

Compute:
- `d_pred = ||center_sam2 - center_pred||`
- `d_tpl  = ||center_sam2 - center_tpl||` (if tpl exists)
- `d_klt  = ||center_sam2 - center_klt||` (if klt exists)
- `border_touch`: mask/bbox touches ROI boundary ratio (0..1)
- optional: `iou_prev`, `area_z` (track-wise stats)

### 7.4 Fusion decision logic (recommended)
- **Case A (Good SAM2)**:
  - `d_pred` small AND (`d_tpl` small if tpl exists) AND `border_touch` low  
  -> accept `center_sam2` as measurement, KF update.
- **Case B (SAM2 suspicious)**:
  - `d_pred` large OR `d_tpl` large OR `border_touch` high  
  -> reject SAM2 for this frame  
  -> use `center_tpl` (preferred) or `center_klt` as measurement  
  -> mark state `UNCERTAIN`, increase miss counters.
- **Case C (All weak)**:
  - tpl/klt not available or low confidence, SAM2 rejected  
  -> no measurement update, KF predict only  
  -> state `OCCLUDED` and accumulate.

### 7.5 Reseed trigger
- `UNCERTAIN/OCCLUDED` persists for `N` frames (e.g., 20–60)
- or detection of obvious ID-swap/jump
-> state `NEEDS_RESEED` and request user correction for that track.

---

## 8. Head Estimation (Secondary, Optional)

### 8.1 Candidate endpoints
- From SAM2 mask:
  - PCA major axis -> projection extremes -> 2 endpoints
  - (optional) skeletonize + find endpoints
- Keep endpoints in global coords (ROI -> full frame).

### 8.2 Selecting the head
- If movement vector `v = center(t) - center(t-1)` is meaningful:
  - choose endpoint more aligned with `v` as head
- If near-stationary / ambiguous:
  - `head=None`
- Optional tie-breaker:
  - endpoint neighborhood activity(flow magnitude) larger end.

> Head accuracy is secondary. Do not force if uncertain.

---

## 9. State Machine (Robust Handling)

### 9.1 States
- `ACTIVE`: 정상 업데이트
- `UNCERTAIN`: 센서 불일치/QA 경고(측정치 제한)
- `OCCLUDED`: 관측 실패(예측만 전진)
- `MERGED`: 트랙 간 겹침으로 분리 어려움
- `EXITED`: arena 밖으로 나가 사라진 것으로 판정
- `DEAD_CANDIDATE`: 장시간 정지 + 기타 조건에서 후보
- `NEEDS_RESEED`: 사용자 재지정 필요

### 9.2 Merge handling
- 겹쳤다고 트랙을 하나로 합치지 않는다(통합 금지).
- `MERGED`로 유지하면서 보수적으로 업데이트(또는 예측+tpl).
- 분리되면 `ACTIVE` 복귀.
- 장기화 시 reseed.

### 9.3 Exit handling
- center가 arena 밖 + 관측 실패 지속 -> `EXITED`.
- EXITED 트랙이 다른 개체를 “대체 인식”하지 않도록, 자동 재탐색 금지(사용자 reseed 시만 복구).

### 9.4 Death handling
- “정지”만으로 사망 확정 금지.
- 후보 상태만 운영(실험 요구에 따라 최종 판정 로직 추가 가능).

---

## 10. Chunking (Memory constraint) — Pass full track state, not just bbox

### 10.1 Past issue
- 청크 경계에서 마지막 bbox만 앵커로 넘기면, bbox가 커진 프레임이 앵커를 망쳐 다음 청크가 붕괴.

### 10.2 New rule
청크 간에는 다음을 반드시 serialize/restore:
- KF state + covariance
- last good center
- template representation (and optionally KLT points)
- quality history / miss count
- last head direction (optional)

### 10.3 Additional note
- JPG 프레임 저장/재로드는 압축 지터를 키움.
- 가능하면 비디오 직접 디코딩 + ROI crop 기반 처리.

---

## 11. Parameters (Initial Defaults)

### ROI
- `roi.base = 384`
- `roi.min = 256`
- `roi.max = 640`

### QA thresholds (tune per fps/scale)
- `qa.dist_pred_thresh = 30~80 px`
- `qa.dist_tpl_thresh  = 30~80 px`
- `qa.border_touch_thresh = 0.3~0.5`
- `qa.uncertain_to_reseed = 20~60 frames`

### KLT (optional)
- points: 50–150
- winSize: 15–21
- maxLevel: 2–3
- reinit if valid points < 10

### Kalman
- state: [x, y, vx, vy]
- process noise q: 1–10 (scale with fps, expected motion)
- measurement noise r: sensor-dependent (SAM2 smallest, tpl medium, klt medium)

---

## 12. Outputs (Required)

### 12.1 `tracks.csv` (or parquet)
Columns recommended:
- `frame_idx, time_sec, track_id`
- `center_x, center_y`
- `head_x, head_y` (NaN if none)
- `state`
- `quality_score`
- `sensor_used` (SAM2/TPL/KLT/PRED)
- `roi_x0, roi_y0, roi_x1, roi_y1`
- debug: `d_pred, d_tpl, d_klt, border_touch, area` etc.

### 12.2 `overlay.mp4`
- arena rectangle
- each track: ROI, center point, head point(if), track_id/state text

### 12.3 `events.jsonl`
- state transitions, merge enter/exit, reseed requested/done, exit detected, etc.

---

## 13. CLI Workflow (Recommended)

1) Arena selection:
- `python select_arena.py --input video.mp4 --out arena.yaml`

2) Seed selection:
- `python select_seeds.py --input video.mp4 --arena arena.yaml --out seeds.yaml`

3) Tracking:
- `python run.py --config config.yaml --input video.mp4 --output output/ts --arena arena.yaml --seeds seeds.yaml`

4) (Optional) Reseed tool for a specific track/time:
- `python reseed_tool.py --input video.mp4 --arena arena.yaml --track 3 --frame 1234 --out seeds_patch.yaml`

---

## 14. Implementation Priority (Suggested Order)

1) `select_seeds.py` 구현(초기 박스 지정 도구) + seeds.yaml 저장/로드
2) ROI crop + Template matching 센서 구현
3) Kalman + 센서 융합(예측/템플릿)
4) SAM2 ROI 센서 연결(좌표만 사용) + QA 게이트
5) 상태머신/이벤트 로그/overlay 정리
6) head 추정(끝점 + 이동 방향)
7) chunk state serialization(필요 시)

---

## 15. Summary of Key Rules (Non-negotiable)

- **SAM2 출력은 “진실”이 아니다.** QA 통과 시에만 채택한다.
- **bbox/area 변화만으로 rollback 금지.** 좌표 일관성 기반으로 판단한다.
- **ROI 무한 확장 금지.** 실패 누적 시 reseed로 해결한다.
- 겹침에서 트랙 통합 금지 → MERGED 상태로 유지.
- 목표는 mask가 아니라 **center/head 좌표의 정확도**다.

---
