# Insect Tracking — Human-in-the-Loop Multi-Sensor Pipeline

실험실 영상에서 **최대 ~10개 애벌레/곤충**을 추적하는 시스템입니다.
사용자가 초기 bounding box(seed)를 지정하면, **SAM2 + Template NCC + KLT Optical Flow** 3개 센서의 교차검증을 통해 프레임별 center/head 좌표를 출력합니다.

---

## Quick Start

```powershell
# 1) 환경 설치
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy opencv-python pyyaml scipy

# SAM2 사용 시 (선택)
pip install torch torchvision
# + sam2/ 서브모듈 설정

# 2) Arena 선택
python select_arena.py --input video.mp4 --out arena.yaml

# 3) Seed 선택 (초기 bbox 지정)
python select_seeds.py --input video.mp4 --arena arena.yaml --out seeds.yaml

# 4) 트래킹 실행
python run.py --config config.yaml --input video.mp4 --output output/run1 --seeds seeds.yaml --arena arena.yaml
```

---

## File Structure

```
insect_tracking/
├── run.py                      # CLI 엔트리포인트
├── config.yaml                 # 전체 파라미터 설정
├── select_arena.py             # Arena 영역 선택 GUI
├── select_seeds.py             # Seed bbox 선택 GUI (프레임 탐색 지원)
├── design_spec.md              # 설계 명세서
├── requirements.txt            # Python 의존성
│
├── pipeline/
│   └── runner.py               # 메인 파이프라인 오케스트레이션
│
├── sensors/                    # 3개 센서 + Head 추정기
│   ├── base.py                 # Sensor ABC + SensorResult dataclass
│   ├── sam2_sensor.py          # SAM2 mask → centroid/PCA endpoints/border_touch
│   ├── template_sensor.py      # Gray+Sobel edge NCC 템플릿 매칭
│   ├── klt_sensor.py           # KLT optical flow (forward-backward 검증)
│   └── head_estimator.py       # PCA endpoints + 이동방향 → head 좌표
│
├── qa/
│   └── fusion.py               # 다중 센서 교차검증 + 융합 (Case A/B/C)
│
├── tracking/
│   ├── track.py                # Track dataclass (7-state, 센서 메모리, 품질)
│   ├── kalman.py               # KalmanFilter2D [x, y, vx, vy]
│   ├── state_machine.py        # 7-state 상태 전이 + merge 감지
│   └── state_io.py             # 청크 간 상태 직렬화/역직렬화
│
├── roi/
│   └── roi_manager.py          # 고정 크기 ROI (Kalman center 기준, arena clamp)
│
├── preprocess/
│   └── preprocess.py           # Grayscale, denoise, optional CLAHE
│
├── io_utils/
│   ├── video_reader.py         # 비디오/이미지 폴더 입력
│   ├── video_writer.py         # 오버레이 영상 저장
│   ├── artifacts.py            # CSV/JSONL 결과 저장 + EventWriter
│   └── overlay.py              # 상태별 색상 오버레이 그리기
│
├── detection/                  # [레거시] 배경차분 기반 자동 탐지
│   ├── background_model.py
│   └── motion_detector.py
│
├── segmentation/               # [레거시] 구 세그멘테이션 인터페이스
│   ├── base.py
│   └── sam2_adapter.py
│
├── tracking/                   # [레거시] Hungarian MOT (현재 미사용)
│   ├── association.py
│   └── multi_tracker.py
│
└── devlog/                     # 작업 기록 (날짜별 md 파일)
    └── 2026-01-27.md
```

---

## Algorithm Flow

### 개요

```
사용자 Seed → 트랙 초기화 → [매 프레임, 트랙별] →
  Kalman Predict → ROI Crop → 3 Sensors → QA Fusion → KF Update →
  State Machine → Head Estimation → Output
```

### Step 1: 사전 준비 (사용자 입력)

1. **Arena 선택** (`select_arena.py`): 영상에서 실험 영역 사각형을 드래그하여 지정. 영역 밖의 트랙은 EXITED 처리됨.
2. **Seed 선택** (`select_seeds.py`): 영상의 원하는 프레임에서 각 애벌레에 bounding box를 드래그하여 지정 (최대 10개). 좌/우 화살표로 프레임 탐색 가능.

### Step 2: 트랙 초기화

- 각 seed bbox의 중심좌표로 **Kalman Filter 2D** 초기화 (`[x, y, vx, vy]` 상태)
- 첫 프레임에서 3개 센서 초기화:
  - **Template**: bbox 영역의 gray + Sobel edge patch 추출
  - **KLT**: bbox 영역에서 `goodFeaturesToTrack` 특징점 초기화
  - **SAM2**: 프레임별 독립 추론 (초기화 불필요)

### Step 3: 매 프레임 Per-Track 루프

각 프레임에서, 활성 트랙마다 다음을 순차 실행:

#### (a) Kalman Predict
- 현재 상태에서 다음 위치 예측 → `pred_center`

#### (b) ROI Crop
- `pred_center` 중심으로 고정 크기 ROI (기본 384x384) 생성
- Arena 영역으로 클램핑

#### (c) 3개 센서 측정

| 센서 | 방법 | 출력 |
|------|------|------|
| **SAM2** | ROI crop에 box prompt → mask 생성 → centroid 계산 | center, PCA endpoints, border_touch, confidence |
| **Template** | ROI 내에서 gray+edge NCC 매칭 → 최대 상관 위치 | center, confidence |
| **KLT** | 이전 프레임 특징점 → optical flow 추적 (forward-backward 검증) → 유효 점 median | center, confidence |

- SAM2는 이전 QA 통과 프레임의 mask center를 **point prompt 힌트**로 사용하여 안정성 향상 (Fix 7)
- SAM2 사용 불가 시 Template + KLT + Kalman만으로 운용

#### (d) QA Fusion (교차검증)

3개 센서 결과를 교차검증하여 최종 좌표와 신뢰도를 결정:

```
d_pred = ||center_SAM2 - pred_center|| / ROI_diagonal
d_tpl  = ||center_SAM2 - center_TPL||  / ROI_diagonal
```

| Case | 조건 | 결과 |
|------|------|------|
| **A (Good SAM2)** | d_pred, d_tpl 모두 임계값 이내 + border_touch 낮음 | SAM2 좌표 채택, ACTIVE |
| **B (Fallback)** | SAM2 불신 또는 MERGED 상태 | Template 또는 KLT 좌표 사용, UNCERTAIN |
| **C (Predict Only)** | 모든 센서 불신 | Kalman 예측값 유지, OCCLUDED |

- MERGED 상태에서는 SAM2를 신뢰하지 않음 (겹친 트랙을 하나로 인식하는 문제 방지)
- 거리 임계값은 ROI diagonal 대비 비율로 정규화 (해상도 독립적)

#### (e) Kalman Update
- Fusion이 승인한 경우에만 관측값으로 Kalman Filter 업데이트
- 센서별 measurement noise 차등 적용 (SAM2: 10, Template/KLT: 25)

#### (f) Track 상태 갱신
- `last_center`, `sensor_used`, `quality_score` 등 업데이트

#### (g) Head Estimation
- **1차**: SAM2 mask의 PCA major axis endpoints에서 이동 방향과 정렬된 쪽을 head로 선택
- **2차 (fallback)**: SAM2 없을 때 ROI crop의 Canny edge PCA로 장축 추정
- 정지 상태이면 head = None

#### (h) State Machine (7-state)

```
ACTIVE ↔ UNCERTAIN ↔ OCCLUDED → NEEDS_RESEED
                                    ↑
MERGED (트랙 겹침 시, 통합 금지)     │
EXITED (arena 밖)                   │
DEAD_CANDIDATE (장기 정지) ─────────┘
```

| 상태 | 의미 | 전이 조건 |
|------|------|-----------|
| ACTIVE | 정상 추적 중 | Fusion Case A 성공 |
| UNCERTAIN | 일부 센서 불일치 | Fusion Case B (miss 누적) |
| OCCLUDED | 모든 센서 실패 | miss_count ≥ uncertain_to_occluded |
| MERGED | 다른 트랙과 겹침 | center 간 거리 < merge_distance |
| EXITED | Arena 밖으로 이탈 | 좌표가 arena 외부 |
| DEAD_CANDIDATE | 장시간 정지 | 평균 속도 < 임계값 (60프레임) |
| NEEDS_RESEED | 사용자 재지정 필요 | miss_count ≥ occluded_to_reseed |

### Step 4: 프레임 간 처리

- **Merge Detection**: 모든 활성 트랙 쌍의 center 거리 비교 → 가까우면 양쪽 MERGED (트랙 통합 금지, 개별 유지)
- **Immobility Check**: 장시간 정지 트랙 → DEAD_CANDIDATE
- **Event Logging**: 상태 전이, MERGE_ENTER/EXIT 이벤트를 `events.jsonl`에 기록

### Step 5: 출력

| 파일 | 내용 |
|------|------|
| `tracks.csv` | frame_idx, time_sec, track_id, center_x/y, head_x/y, state, quality_score, sensor_used, bbox, area |
| `events.jsonl` | 상태 전이, merge 이벤트 |
| `overlay.mp4` | Arena, bbox, center(원), head(화살표), track_id+state 텍스트 (상태별 색상) |

---

## Key Design Principles

1. **SAM2 출력 ≠ 정답** — QA 교차검증 통과 시에만 채택
2. **bbox/area 변화만으로 rollback 금지** — 좌표 일관성 기반 판단
3. **ROI 무한 확장 금지** — 실패 누적 시 NEEDS_RESEED
4. **트랙 통합 금지** — MERGED 상태로 개별 유지
5. **목표는 center/head 좌표 정확도** (mask 품질 자체가 아님)
6. **사용자 seed 필수** — 자동 새 트랙 생성 없음

---

## Config

`config.yaml`에서 주요 파라미터 조정 가능:

| 섹션 | 주요 항목 |
|------|-----------|
| `roi` | base_size (384), min/max_size |
| `sensors.sam2` | enabled, checkpoint, measurement_noise_r |
| `sensors.template` | patch_size, update_quality_thresh |
| `sensors.klt` | max_corners, min_valid_points |
| `qa` | dist_pred_ratio, dist_tpl_ratio, border_touch_thresh |
| `tracking.kalman` | process_noise_q, measurement_noise_r |
| `tracking.state_machine` | merge_distance_px, immobile_window |

---

## Development Log

작업 기록은 `devlog/` 폴더에 날짜별 markdown 파일로 관리합니다.
