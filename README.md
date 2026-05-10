# 당신과 가장 닮은 동물은? — Stage 2/3 baseline

사람 얼굴 사진과 동물 종 텍스트를 입력 받아, 그 종의 모습으로 합성된 단일 이미지를 생성하는 두 가지 conditioning 설계를 비교 실험한다.

| | Stage 2 | Stage 3 |
|---|---|---|
| identity | IP-Adapter-FaceID (ArcFace) | IP-Adapter-FaceID (ArcFace) |
| 공간 구조 | 없음 | ControlNet (MediaPipe FaceMesh) |

## 빠른 시작

```bash
# 의존성 설치 (단일 GPU 12-16GB 환경 가정, SD 1.5)
pip install -r requirements.txt

# 모델 가중치 다운로드
python scripts/setup_models.py

# 본인 사진을 data/faces/raw/ 에 넣고
python scripts/prepare_faces.py --in data/faces/raw --out data/faces/aligned

# 스모크 테스트 (조합 1개씩만)
python scripts/run_sweep.py --config configs/stage2_ipadapter.yaml --limit 1
python scripts/run_sweep.py --config configs/stage3_pose.yaml      --limit 1

# 본 sweep
python scripts/run_sweep.py --config configs/stage2_ipadapter.yaml
python scripts/run_sweep.py --config configs/stage3_pose.yaml
```

## 산출물

각 sweep은 `experiments/runs/<stage>_<timestamp>/` 아래에:
- `config.yaml` — 사용된 config 스냅샷
- `images/<face_id>__<species>__ip<scale>[__cn<scale>]__seed<n>.png`
- `metadata.parquet` — 한 줄 = 한 생성, 모든 hyperparam 포함

평가는 노트북에서 격자로 정렬해 사람이 직접 본다 (`notebooks/0[1-3]_*.ipynb`).

## 참고

- Base diffusion: Stable Diffusion 1.5
- IP-Adapter: `h94/IP-Adapter-FaceID` (`ip-adapter-faceid_sd15.bin`)
- ControlNet: `CrucibleAI/ControlNetMediaPipeFace` (1차) / `lllyasviel/control_v11p_sd15_openpose` (fallback)
- Face: insightface `buffalo_l` (ArcFace + 5-keypoint 정렬)
