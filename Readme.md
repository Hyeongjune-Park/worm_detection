# Insect Tracking (Larva) – ROI/MOT Pipeline (No-Training)

실험실 영상에서 **최대 ~10개 애벌레**를 라벨링/학습 없이 추적하기 위한 코드베이스입니다.  
핵심 목표는 “어떤 영상이 오더라도” 배경(풀잎/물결/반사/자막 등)로 빨려 들어가는 문제를 줄이고, 유지보수 가능한 구조로 확장하는 것입니다.

> ✅ 현재 파이프라인은 **MOT(다중 객체 추적) 중심**으로 동작합니다.  
> ✅ SAM2는 “선택적 보정(ROI 세그멘테이션)”로 붙이는 구조이며, 연결이 안 되면 자동으로 fallback 세그멘터로 동작합니다.

---

## TL;DR (빠른 시작)

### 1) 가상환경 + 설치
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install numpy opencv-python pyyaml scipy


insect_tracking/
├─ run.py                         # CLI 엔트리. config + arena yaml 로드 후 파이프라인 실행
├─ config.yaml                    # 알고리즘 파라미터(검출/트래킹/ROI/SAM2 설정)
├─ select_arena.py                # 수동 사각형 아레나 선택기(드래그 → Enter/S 저장)
│
├─ video/                         # 입력 영상 폴더(예: short.mp4)
├─ output/                        # 실행 결과 저장(타임스탬프 폴더 권장)
├─ sam2/                          # SAM2 코드/체크포인트(옵션)
│
├─ pipeline/
│  ├─ __init__.py                 # (비워둠)
│  └─ runner.py                   # 전체 흐름 조립(Reader → Preprocess → Detect → Track → Save)
│
├─ preprocess/
│  ├─ __init__.py                 # (비워둠)
│  └─ preprocess.py               # 프레임 전처리(denoise/gray/normalize 등)
│
├─ detection/
│  ├─ __init__.py                 # (비워둠)
│  ├─ background_model.py         # 배경 모델(RunningAvgBackground)
│  └─ motion_detector.py          # 모션 기반 fg/blob + (선택) 아레나 마스크 적용
│
├─ tracking/
│  ├─ __init__.py                 # (비워둠)
│  ├─ track.py                    # Track 데이터 구조(상태/마지막 관측/품질 등)
│  ├─ kalman.py                   # 2D 칼만필터(예측 중심점)
│  ├─ association.py              # 데이터 연계(Hungarian/greedy + 게이팅)
│  ├─ state_machine.py            # ACTIVE/OCCLUDED/EXITED/DEAD 등 상태 전이
│  └─ multi_tracker.py            # MOT 본체(트랙 생성/업데이트/유실 관리)
│
├─ roi/
│  ├─ __init__.py                 # (비워둠)
│  └─ roi_manager.py              # 트랙별 ROI 산출(예측 중심 + padding + 확장 규칙)
│
├─ segmentation/
│  ├─ __init__.py                 # (반드시 비워두기: 패키지 마커)
│  ├─ base.py                     # Segmenter 인터페이스
│  ├─ sam2_adapter.py             # SAM2 ROI 세그멘테이션 어댑터(옵션, 실패 시 fallback)
│  └─ quality.py                  # 마스크 품질/일관성 지표(선택)
│
└─ io_utils/
   ├─ __init__.py                 # (비워둠)
   ├─ video_reader.py             # 비디오/프레임 폴더 입력 스트리밍
   ├─ video_writer.py             # 오버레이 영상 저장(선택)
   └─ artifacts.py                # 결과 저장(tracks.jsonl/csv 등)
