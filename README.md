# sDCTFT Fine-Tuning for SDXL

SDXL 모델을 위한 **Selective DCT Fine-Tuning** 구현체입니다.
논문: [arXiv:2410.09103](https://arxiv.org/abs/2410.09103) (MaCP / sDCTFT)

---

## 개요

sDCTFT는 가중치 행렬에 2D DCT를 적용하고 주파수 도메인의 일부 계수만 학습하는 파라미터 효율적 파인튜닝 방법입니다.
SDXL UNet 2.57B 파라미터 중 **84,560개(0.003%)** 만 학습하며, 결과물은 kohya 호환 LoRA safetensors로 저장됩니다.

![이미지 스펙트럼 예시](https://github.com/bemoregt/Train_sDCTFT_FineTuning_using_SDXLModel/blob/main/ScrShot%2024.png)

### 알고리즘 (논문 Algorithm 1)

1. `W ∈ R^{d_out × d_in}`에 2D DCT-II 적용: `W_F = C_out @ W @ C_in.T`
2. 주파수 거리 `d(u,v) = √(u²+v²)` 기준 3 밴드 분할
3. 각 밴드: `|W_F|` 에너지 상위 70% + 랜덤 30% 선택 (`n_per_band`개씩)
4. 선택 위치만 Kaiming 초기화 → 학습 파라미터 `delta_vals`
5. Forward (factored 2-hop, ΔW 미실체화):
   ```
   h = x @ C_rows_in.T          # [..., n_sel]
   h = h * delta_vals            # [..., n_sel]
   Δy = (h @ C_rows_out) * α    # [..., d_out]
   ```
6. 학습 후 SVD 압축 → kohya LoRA safetensors → ComfyUI 바로 사용

---

## 요구사항

```
torch>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0
safetensors>=0.3.0
diffusers
transformers
scipy
torchvision
```

```bash
pip install -r requirements.txt
pip install diffusers transformers scipy torchvision
```

---

## 사용법

### 기본 실행

```bash
python finetune_sdctft.py
```

- `images/` 폴더의 이미지로 학습
- SDXL 베이스 모델 자동 다운로드 (최초 1회)
- 결과물: `output_sdctft/` 에 LoRA safetensors 저장

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `images` | 학습 이미지 폴더 |
| `--model_path` | None | 로컬 SDXL 모델 경로 (없으면 HuggingFace 다운로드) |
| `--output_dir` | `output_sdctft` | 출력 폴더 |
| `--image_size` | 512 | 학습 해상도 (M1은 256 권장) |
| `--epochs` | 20 | 학습 에폭 수 |
| `--lr` | 5e-5 | 학습률 |
| `--batch_size` | 1 | 배치 크기 |
| `--grad_accum` | 4 | 그래디언트 누적 (유효 배치 = batch_size × grad_accum) |
| `--n_per_band` | 50 | 밴드당 선택 주파수 수 |
| `--energy_ratio` | 0.7 | 에너지 기반 선택 비율 (나머지는 랜덤) |
| `--lora_rank` | 16 | LoRA 저장 시 SVD 랭크 |
| `--save_every` | 5 | 에폭마다 중간 체크포인트 저장 주기 |

### 예시

```bash
# 로컬 모델 사용, 30에폭, M1 최적화 (256 해상도)
python finetune_sdctft.py \
    --model_path ./models/sd_xl_base_1.0.safetensors \
    --epochs 30 \
    --image_size 256 \
    --lr 5e-5

# 고해상도, 더 많은 주파수 선택
python finetune_sdctft.py \
    --image_size 512 \
    --n_per_band 100 \
    --lora_rank 32
```

---

## 출력

```
output_sdctft/
├── <name>_ep5.safetensors    # 중간 체크포인트
├── <name>_ep10.safetensors
├── ...
└── <name>_sdctft_final.safetensors  # 최종 LoRA
```

---

## ComfyUI 사용법

1. `output_sdctft/*.safetensors` → ComfyUI `models/loras/` 폴더에 복사
2. 워크플로우 구성:
   ```
   CheckpointLoaderSimple (SDXL 베이스)
           ↓ model / clip
   LoraLoader (pierr_probst_sdctft_final, strength 0.8)
           ↓ model / clip
   CLIP 텍스트 인코더 + KSampler
   ```
3. 프롬프트에 학습 시 사용한 스타일 키워드 포함

---

## Apple Silicon (MPS) 참고

- `image_size=256` 권장 (512는 ~30s/step으로 매우 느림)
- 처음 약 30 스텝은 Metal 셰이더 JIT 컴파일로 느림 → 이후 ~1.2s/step
- VAE는 fp16 NaN 방지를 위해 자동으로 fp32 전환
- GroupNorm도 fp32 hook 자동 적용
