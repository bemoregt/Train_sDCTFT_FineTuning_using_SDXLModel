"""
sDCTFT (Selective DCT Fine-Tuning) for SDXL — Pierr Probst 스타일
==================================================================
논문: "Parameter-Efficient Fine-Tuning via Selective Discrete Cosine Transform"
      arXiv:2410.09103 (MaCP / sDCTFT)

알고리즘 (논문 Algorithm 1):
  1. W ∈ R^{d_out×d_in}에 2D DCT-II 적용: W_F = C_out @ W @ C_in.T
  2. 주파수 거리 d(u,v)=√(u²+v²) 기준 3 밴드 분할
  3. 각 밴드: |W_F| 에너지 상위 70% + 랜덤 30% 선택 (n_per_band개씩)
  4. 선택 위치만 Kaiming 초기화 → 학습 파라미터 delta_vals
  5. Forward (순수 PyTorch, autograd 완전 지원):
       ΔW = C_out[:,sel_r].T  @  diag(delta_vals)  @  C_in[sel_c,:]  ×  α
       W_eff = W_base + ΔW
  6. 학습 후 SVD 압축 → kohya LoRA safetensors → ComfyUI 바로 사용

수학적 근거:
  iDCT2D(ΔW_F)[a,b] = Σ_i delta_vals[i] · C_out[sel_r_i, a] · C_in[sel_c_i, b]
                     = C_out[:,sel_r].T @ diag(delta_vals) @ C_in[sel_c,:]  [a,b]
  → 완전 미분 가능, device-agnostic, 효율적 O(n_sel × max(d_out,d_in))

사용법:
    python finetune_sdctft.py                      # SDXL 자동 다운로드
    python finetune_sdctft.py --epochs 30 --lr 5e-5
    python finetune_sdctft.py --model_path ./models/sd_xl_base_1.0.safetensors
"""

import math
import os
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# DCT 행렬 & 주파수 선택 (초기화 전용)
# ---------------------------------------------------------------------------

def _make_dct_matrix_np(N: int):
    """
    N×N 정규직교 DCT-II 행렬 (numpy).
    C[k, n] = α(k) · cos(π(n+0.5)k/N)
    """
    import numpy as np
    k = np.arange(N)[:, None]   # [N, 1]
    n = np.arange(N)[None, :]   # [1, N]
    C = np.cos(np.pi * k * (n + 0.5) / N)
    C[0] *= 1.0 / math.sqrt(N)
    C[1:] *= math.sqrt(2.0 / N)
    return C.astype(np.float64)   # [N, N]


def _dct2d_np(W_np):
    """2D DCT-II (numpy, 초기화 전용 에너지 계산용)."""
    from scipy.fft import dctn
    return dctn(W_np.astype(np.float64), norm="ortho")


def _select_dct_indices(W_F_np, n_per_band: int = 50, energy_ratio: float = 0.7):
    """
    논문 Algorithm 1: 3 주파수 밴드별 선택.
    Returns: (sel_r, sel_c) — 각각 1D int64 numpy array
    """
    import numpy as np
    d_out, d_in = W_F_np.shape
    d_max = math.sqrt((d_out / 2) ** 2 + (d_in / 2) ** 2)

    u = np.arange(d_out)[:, None]
    v = np.arange(d_in)[None, :]
    dist = np.sqrt(u ** 2 + v ** 2)

    bands = [
        dist <= d_max / 3,
        (dist > d_max / 3) & (dist <= 2 * d_max / 3),
        dist > 2 * d_max / 3,
    ]

    selected_r, selected_c = [], []
    abs_F = np.abs(W_F_np)

    for band in bands:
        idx = np.argwhere(band)   # [N_band, 2]
        N_band = len(idx)
        if N_band == 0:
            continue
        n_sel = min(n_per_band, N_band)
        n_energy = max(1, int(n_sel * energy_ratio))
        n_rand = n_sel - n_energy

        vals = abs_F[band]
        top_local = np.argsort(vals)[::-1][:n_energy]
        chosen = set(top_local.tolist())

        remaining = [i for i in range(N_band) if i not in chosen]
        if remaining and n_rand > 0:
            extra = random.sample(remaining, min(n_rand, len(remaining)))
            chosen.update(extra)

        for i in chosen:
            selected_r.append(idx[i, 0])
            selected_c.append(idx[i, 1])

    return (np.array(selected_r, dtype=np.int64),
            np.array(selected_c, dtype=np.int64))


import numpy as np   # needed module-level after _select_dct_indices

# DCT 행렬 캐시: 동일 차원에 대해 반복 계산 방지 (560 레이어 초기화 속도 향상)
_DCT_MATRIX_CACHE: dict[int, np.ndarray] = {}

def _make_dct_matrix_np_cached(N: int) -> np.ndarray:
    """캐시된 DCT-II 행렬. 동일 N은 한 번만 계산."""
    if N not in _DCT_MATRIX_CACHE:
        _DCT_MATRIX_CACHE[N] = _make_dct_matrix_np(N)
    return _DCT_MATRIX_CACHE[N]


# ---------------------------------------------------------------------------
# sDCTFT Linear Layer  (autograd 완전 지원)
# ---------------------------------------------------------------------------

class sDCTFTLinear(nn.Module):
    """
    nn.Linear → sDCTFT 어댑터.

    Forward (미분 가능):
        ΔW = C_rows_out.T @ (delta_vals[:,None] * C_rows_in) * alpha
        W_eff = W_base + ΔW

    여기서:
        C_rows_out[i,:] = C_out[sel_r[i],:]  (선택된 DCT 기저 행, 출력 차원)
        C_rows_in[i,:]  = C_in[sel_c[i],:]   (선택된 DCT 기저 행, 입력 차원)
    """

    def __init__(
        self,
        linear: nn.Linear,
        n_per_band: int = 50,
        energy_ratio: float = 0.7,
    ):
        super().__init__()
        W = linear.weight.data.float().cpu()   # [d_out, d_in]
        d_out, d_in = W.shape
        self.d_out, self.d_in = d_out, d_in
        self.has_bias = linear.bias is not None

        # 원본 가중치 동결 버퍼 (fp16으로 저장 → 메모리 절약)
        self.register_buffer("W_base", W.half())
        if self.has_bias:
            self.register_buffer("bias_buf", linear.bias.data.float().cpu())
        else:
            self.bias_buf = None

        # --- 초기화 전용: DCT + 주파수 선택 ---
        W_F_np = _dct2d_np(W.numpy())
        sel_r, sel_c = _select_dct_indices(W_F_np, n_per_band, energy_ratio)
        n_sel = len(sel_r)

        # DCT 기저 행 추출: C_out[sel_r,:] and C_in[sel_c,:]  (캐시 사용)
        C_out_np = _make_dct_matrix_np_cached(d_out)   # [d_out, d_out]
        C_in_np  = _make_dct_matrix_np_cached(d_in)    # [d_in,  d_in]

        # fp16으로 저장 → 메모리 절약 (560 레이어 × ~0.5 MB = ~280 MB)
        C_rows_out = torch.from_numpy(C_out_np[sel_r, :]).half()   # [n_sel, d_out]
        C_rows_in  = torch.from_numpy(C_in_np[sel_c, :]).half()    # [n_sel, d_in]

        self.register_buffer("C_rows_out", C_rows_out)  # 고정
        self.register_buffer("C_rows_in",  C_rows_in)   # 고정

        # 학습 파라미터: 선택된 DCT 계수 (Kaiming 초기화)
        self.delta_vals = nn.Parameter(torch.empty(n_sel))
        nn.init.kaiming_uniform_(self.delta_vals.unsqueeze(0), a=math.sqrt(5))
        self.delta_vals.data = self.delta_vals.data.squeeze(0)

        # 스케일 α (논문의 scaling factor)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def _compute_delta_W(self) -> torch.Tensor:
        """
        ΔW = C_rows_out.T @ diag(delta_vals) @ C_rows_in × α
           = [d_out, n_sel] @ [n_sel, d_in] × α
        완전 미분 가능, device-agnostic.
        fp32 정밀도로 계산 (gradient 안정성).
        """
        # scaled: [n_sel, d_in] — fp32로 계산
        scaled = self.delta_vals.float().unsqueeze(1) * self.C_rows_in.float()
        # delta_W: [d_out, d_in]
        delta_W = self.C_rows_out.float().T @ scaled
        return delta_W * self.alpha.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 효율적 분해: ΔW를 메모리에 구체화하지 않고 2-hop matmul로 계산
        # 메모리: [d_out, d_in] 대신 [B, seq, n_sel] 중간 텐서만 저장
        # x: [..., d_in]
        bias = self.bias_buf.to(x.dtype) if self.has_bias else None
        y_base = F.linear(x, self.W_base.to(x.dtype), bias)

        # delta path (fp32 for gradient stability):
        # h = x @ C_rows_in.T → [..., n_sel]
        # delta = h * delta_vals → [..., n_sel]
        # out  = delta @ C_rows_out → [..., d_out]
        x_f = x.float()
        h = x_f @ self.C_rows_in.float().t()          # [..., n_sel]
        h = h * self.delta_vals.float()                # [..., n_sel]
        delta = (h @ self.C_rows_out.float()) * self.alpha.float()  # [..., d_out]
        return y_base + delta.to(x.dtype)

    @torch.no_grad()
    def get_merged_weight(self) -> torch.Tensor:
        """병합된 가중치 W_base + ΔW (float32)."""
        return self.W_base.float() + self._compute_delta_W().float()

    @torch.no_grad()
    def get_delta_W(self) -> torch.Tensor:
        """ΔW만 반환 (float32, LoRA 저장용)."""
        return self._compute_delta_W().float()


# ---------------------------------------------------------------------------
# UNet에 sDCTFT 설치
# ---------------------------------------------------------------------------

def apply_sdctft_to_unet(unet, n_per_band: int = 50, energy_ratio: float = 0.7):
    """
    UNet의 모든 Attention projection Linear → sDCTFTLinear 교체.
    대상: to_q, to_k, to_v, to_out[0]
    """
    from diffusers.models.attention_processor import Attention as DiffAttn

    replaced = 0
    for name, module in unet.named_modules():
        if not isinstance(module, DiffAttn):
            continue
        for attr in ("to_q", "to_k", "to_v"):
            lin = getattr(module, attr, None)
            if isinstance(lin, nn.Linear):
                setattr(module, attr,
                        sDCTFTLinear(lin, n_per_band, energy_ratio))
                replaced += 1
        if hasattr(module, "to_out") and len(module.to_out) > 0:
            if isinstance(module.to_out[0], nn.Linear):
                module.to_out[0] = sDCTFTLinear(
                    module.to_out[0], n_per_band, energy_ratio)
                replaced += 1

    # 백본 동결: sDCTFT 파라미터만 학습 가능
    for name, param in unet.named_parameters():
        if "delta_vals" not in name and "alpha" not in name:
            param.requires_grad_(False)

    print(f"[sDCTFT] {replaced}개 Linear → sDCTFTLinear 교체 완료")
    return unet


# ---------------------------------------------------------------------------
# SDXL UNet 키 매핑 (diffusers ↔ LDM/kohya)
# ---------------------------------------------------------------------------

_DIFFUSERS_TO_LDM = {
    "down_blocks.1.attentions.0.": "model.diffusion_model.input_blocks.4.1.",
    "down_blocks.1.attentions.1.": "model.diffusion_model.input_blocks.5.1.",
    "down_blocks.2.attentions.0.": "model.diffusion_model.input_blocks.7.1.",
    "down_blocks.2.attentions.1.": "model.diffusion_model.input_blocks.8.1.",
    "mid_block.attentions.0.":     "model.diffusion_model.middle_block.1.",
    "up_blocks.0.attentions.0.":   "model.diffusion_model.output_blocks.0.1.",
    "up_blocks.0.attentions.1.":   "model.diffusion_model.output_blocks.1.1.",
    "up_blocks.0.attentions.2.":   "model.diffusion_model.output_blocks.2.1.",
    "up_blocks.1.attentions.0.":   "model.diffusion_model.output_blocks.3.1.",
    "up_blocks.1.attentions.1.":   "model.diffusion_model.output_blocks.4.1.",
    "up_blocks.1.attentions.2.":   "model.diffusion_model.output_blocks.5.1.",
}


def diffusers_to_kohya_key(diffusers_key: str):
    """
    diffusers UNet 모듈 이름 → kohya LoRA 키.
    예) 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q'
      → 'lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q'
    """
    for d_pfx, ldm_pfx in _DIFFUSERS_TO_LDM.items():
        if diffusers_key.startswith(d_pfx):
            ldm_key = ldm_pfx + diffusers_key[len(d_pfx):]
            # 'model.diffusion_model.' 제거 후 '.' → '_', lora_unet_ 접두사
            inner = ldm_key.replace("model.diffusion_model.", "")
            kohya = "lora_unet_" + inner.replace(".", "_")
            return kohya
    return None


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

class StyleDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, size: int = 512):
        self.paths = [
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in self.EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"이미지 없음: {root}")
        print(f"[Dataset] {len(self.paths)}개 이미지 로드")

        self.transform = transforms.Compose([
            transforms.Resize(
                int(size * 1.05),
                interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        except Exception:
            return self[(idx + 1) % len(self)]


# ---------------------------------------------------------------------------
# SDXL 텍스트 인코딩
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_sdxl(tokenizer1, tokenizer2,
                        text_encoder1, text_encoder2,
                        prompt: str, device, dtype):
    """
    SDXL dual text encoding.
    Returns: prompt_embeds [1, 77, 2048], pooled [1, 1280]
    """
    def _tok_enc(tokenizer, encoder, out_idx=-2):
        toks = tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        out = encoder(toks.input_ids.to(device), output_hidden_states=True)
        return out.hidden_states[out_idx], getattr(out, "pooler_output", None)

    enc1_hidden, _       = _tok_enc(tokenizer1, text_encoder1)   # [1, 77, 768]
    enc2_hidden, pooled2 = _tok_enc(tokenizer2, text_encoder2)   # [1, 77, 1280]

    # pooled_prompt_embeds: OpenCLIP-G last token
    if pooled2 is None:
        with torch.no_grad():
            toks2 = tokenizer2(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            )
            enc2_out = text_encoder2(
                toks2.input_ids.to(device), output_hidden_states=True
            )
            # 마지막 token이 pooled embedding
            pooled2 = enc2_out[0]

    prompt_embeds = torch.cat([enc1_hidden, enc2_hidden], dim=-1).to(dtype)
    pooled_embeds = pooled2.to(dtype)
    return prompt_embeds, pooled_embeds


# ---------------------------------------------------------------------------
# GroupNorm fp32 래퍼 (MPS fp16 NaN 방지)
# ---------------------------------------------------------------------------

def make_groupnorm_fp32(module) -> None:
    """
    MPS에서 fp16 GroupNorm은 분산 계산 중 NaN 발생.
    모든 GroupNorm을 fp32 파라미터로 변환하고, forward hook으로
    입력을 fp32로 업캐스트, 출력을 원래 dtype으로 다시 변환.
    GroupNorm 파라미터 크기는 작아 메모리 영향 미미.

    Note: post_hook의 args는 pre_hook 이후 변환된 입력(fp32)이므로,
    원래 dtype을 저장하기 위해 per-layer state dict 사용.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, torch.nn.GroupNorm):
            m.float()
            # per-layer 클로저로 원래 dtype 캡처
            state = {'orig_dtype': None}

            def _pre(mod, args, _s=state):
                if args and isinstance(args[0], torch.Tensor):
                    _s['orig_dtype'] = args[0].dtype
                return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)

            def _post(mod, args, output, _s=state):
                if _s['orig_dtype'] is not None and isinstance(output, torch.Tensor):
                    return output.to(_s['orig_dtype'])
                return output

            m.register_forward_pre_hook(_pre)
            m.register_forward_hook(_post)
            count += 1
    print(f"[fp32] GroupNorm {count}개 fp32 전환 완료")


# ---------------------------------------------------------------------------
# VAE 사전 인코딩 (학습 전 1회, 매 스텝마다 VAE forward 불필요)
# ---------------------------------------------------------------------------

def precompute_latents(vae, dataset, device, dtype, batch_size=4):
    """
    모든 이미지를 VAE로 한 번 인코딩해 latent 텐서 리스트 반환.
    학습 중 VAE를 device에 올리지 않아도 돼 메모리 절약 + 속도 향상.
    """
    from torch.utils.data import DataLoader as _DL
    loader = _DL(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vae.to(device)
    latents = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            lat = vae.encode(imgs.float()).latent_dist.sample()
            lat = (lat * vae.config.scaling_factor).to(dtype)
            latents.append(lat.cpu())
    vae.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    all_lat = torch.cat(latents, dim=0)  # [N, 4, H/8, W/8]
    print(f"[VAE] 사전 인코딩 완료: {all_lat.shape}, finite={torch.isfinite(all_lat).all().item()}")
    return all_lat


# ---------------------------------------------------------------------------
# 학습 스텝
# ---------------------------------------------------------------------------

def train_step(
    unet, noise_scheduler, optimizer,
    latent, prompt_embeds, pooled_embeds, add_time_ids,
    device, dtype, grad_accum: int, step_idx: int,
) -> float:
    # latent는 사전 인코딩된 텐서 (CPU에서 전달, device로 이동)
    latent = latent.to(device, dtype=dtype)
    B = latent.shape[0]

    t = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (B,), device=device
    ).long()
    noise = torch.randn_like(latent)
    noisy = noise_scheduler.add_noise(latent, noise, t)

    added_cond = {
        "text_embeds": pooled_embeds.expand(B, -1),
        "time_ids":    add_time_ids.expand(B, -1),
    }

    pred = unet(
        noisy, t,
        encoder_hidden_states=prompt_embeds.expand(B, -1, -1),
        added_cond_kwargs=added_cond,
    ).sample

    # NaN 체크 후 skip (안전장치)
    if not torch.isfinite(pred).all():
        optimizer.zero_grad()
        return 0.0

    loss = F.mse_loss(pred, noise) / grad_accum
    loss.backward()

    if (step_idx + 1) % grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in unet.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * grad_accum


# ---------------------------------------------------------------------------
# kohya LoRA 저장 (ComfyUI 호환)
# ---------------------------------------------------------------------------

def save_as_lora(unet, output_path: str, lora_rank: int = 16):
    """
    sDCTFT ΔW → SVD 압축 → kohya LoRA safetensors.
    ComfyUI에서 base SDXL + 이 LoRA로 Pierr Probst 스타일 생성 가능.
    """
    from safetensors.torch import save_file

    lora_state = {}
    saved = skipped = 0

    for name, module in unet.named_modules():
        if not isinstance(module, sDCTFTLinear):
            continue

        delta_W = module.get_delta_W()   # [d_out, d_in], float32, CPU

        # SVD 분해
        try:
            U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        except Exception:
            skipped += 1
            continue

        rank = min(lora_rank, len(S))
        eff_rank = int((S[:rank] > 1e-6).sum().item())
        if eff_rank == 0:
            skipped += 1
            continue
        rank = max(1, eff_rank)

        # lora_down [rank, d_in], lora_up [d_out, rank]
        sqrt_S = S[:rank].sqrt()
        lora_down = (Vt[:rank] * sqrt_S.unsqueeze(1)).to(torch.float16).contiguous()
        lora_up   = (U[:, :rank] * sqrt_S.unsqueeze(0)).to(torch.float16).contiguous()

        kohya_key = diffusers_to_kohya_key(name)
        if kohya_key is None:
            skipped += 1
            continue

        lora_state[f"{kohya_key}.lora_down.weight"] = lora_down
        lora_state[f"{kohya_key}.lora_up.weight"]   = lora_up
        lora_state[f"{kohya_key}.alpha"] = torch.tensor(
            float(rank), dtype=torch.float32
        )
        saved += 1

    print(f"[LoRA] {saved}개 저장, {skipped}개 스킵")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(lora_state, output_path)
    print(f"[LoRA] → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# SDXL 다운로드
# ---------------------------------------------------------------------------

def download_sdxl(save_dir: str = "./models") -> str:
    from huggingface_hub import hf_hub_download
    import shutil

    os.makedirs(save_dir, exist_ok=True)
    local = os.path.join(save_dir, "sd_xl_base_1.0.safetensors")

    if os.path.exists(local):
        gb = os.path.getsize(local) / 1e9
        print(f"[SDXL] 기존 파일 사용: {local}  ({gb:.1f} GB)")
        return local

    print("[SDXL] 다운로드: stabilityai/stable-diffusion-xl-base-1.0  (~6.9 GB)")
    cached = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        local_dir=save_dir,
    )

    comfy_dir = Path.home() / "ComfyUI" / "models" / "checkpoints"
    if comfy_dir.exists():
        dst = comfy_dir / "sd_xl_base_1.0.safetensors"
        if not dst.exists():
            try:
                os.symlink(os.path.abspath(cached), dst)
            except OSError:
                shutil.copy2(cached, dst)
            print(f"[SDXL] ComfyUI checkpoints에 추가: {dst}")

    print(f"[SDXL] 완료: {cached}")
    return cached


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

STYLE_PROMPT = "Pierr Probst style"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="images")
    parser.add_argument("--model_path",   default=None)
    parser.add_argument("--output_dir",   default="output_sdctft")
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--n_per_band",   type=int,   default=50)
    parser.add_argument("--energy_ratio", type=float, default=0.7)
    parser.add_argument("--lora_rank",    type=int,   default=16)
    parser.add_argument("--save_every",   type=int,   default=5)
    args = parser.parse_args()

    # 장치
    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32
    print(f"장치: {device}  dtype: {dtype}")

    # 데이터 경로
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = str(Path(__file__).parent / data_dir)

    # ── 1. SDXL 다운로드 ────────────────────────────────────────────────────
    model_path = args.model_path or download_sdxl(
        str(Path(__file__).parent / "models")
    )

    # ── 2. 파이프라인 로드 ──────────────────────────────────────────────────
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler

    print(f"\n[Load] {model_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    unet = pipe.unet
    vae  = pipe.vae
    text_encoder1 = pipe.text_encoder
    text_encoder2 = pipe.text_encoder_2
    tokenizer1    = pipe.tokenizer
    tokenizer2    = pipe.tokenizer_2

    vae.eval().requires_grad_(False)
    text_encoder1.eval().requires_grad_(False)
    text_encoder2.eval().requires_grad_(False)

    # MPS fp16 NaN 방지: VAE 전체를 fp32로 실행 (크기 ~0.6 GB, 허용 가능)
    # VAE는 학습하지 않으므로 fp32 메모리 영향 최소
    if device.type == "mps":
        vae.float()
        print("[fp32] VAE fp32 변환 완료")

    total = sum(p.numel() for p in unet.parameters())
    print(f"UNet 파라미터: {total:,}")

    # ── 3. sDCTFT 설치 ──────────────────────────────────────────────────────
    print(f"\n[sDCTFT] 설치 중  n_per_band={args.n_per_band} "
          f"energy_ratio={args.energy_ratio} ...")
    unet = apply_sdctft_to_unet(unet, args.n_per_band, args.energy_ratio)
    # 어댑터 설치 후 전체 UNet을 다시 device로 이동 (새 레이어가 CPU에 있을 수 있음)
    unet = unet.to(device)
    # MPS fp16 NaN 방지: GroupNorm을 fp32로 실행
    if device.type == "mps":
        make_groupnorm_fp32(unet)
    # NOTE: gradient_checkpointing은 MPS에서 불안정 → 비활성화
    # unet.enable_gradient_checkpointing()

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"학습 파라미터: {trainable:,}  ({100*trainable/total:.3f}%)\n")

    # ── 4. 스케줄러 & 텍스트 임베딩 캐시 ────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    print(f'[Text] 프롬프트: "{STYLE_PROMPT}"')
    prompt_embeds, pooled_embeds = encode_prompt_sdxl(
        tokenizer1, tokenizer2,
        text_encoder1, text_encoder2,
        STYLE_PROMPT, device, dtype,
    )
    # 텍스트 인코더를 CPU로 오프로드 → MPS 메모리 ~2.6 GB 확보
    text_encoder1.to("cpu")
    text_encoder2.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    print("[Memory] 텍스트 인코더 CPU 오프로드 완료")

    sz = args.image_size
    add_time_ids = torch.tensor(
        [[sz, sz, 0, 0, sz, sz]], dtype=dtype, device=device
    )

    # ── 5. 데이터셋 & VAE 사전 인코딩 ─────────────────────────────────────
    dataset = StyleDataset(data_dir, size=sz)
    print(f"[Dataset] {len(dataset)}개 이미지 로드")

    # VAE를 한 번만 실행해 latent 캐시 생성 → 학습 중 VAE 불필요
    all_latents = precompute_latents(vae, dataset, device, dtype, batch_size=4)
    # TensorDataset으로 교체
    from torch.utils.data import TensorDataset
    latent_dataset = TensorDataset(all_latents)
    loader = DataLoader(latent_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=False)
    # VAE는 더 이상 필요 없음 — CPU에 유지 (메모리 절약)
    vae.to("cpu")

    # ── 6. 옵티마이저 ───────────────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 7. 학습 루프 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" sDCTFT SDXL 파인튜닝 — Pierr Probst 스타일")
    print(f" 이미지: {len(dataset)}장  에폭: {args.epochs}  "
          f"유효배치: {args.batch_size * args.grad_accum}")
    print(f"{'='*60}\n")

    unet.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        total_loss = 0.0
        log_every  = max(1, len(loader) // 5)

        for step_idx, (latent,) in enumerate(loader):
            loss = train_step(
                unet, noise_scheduler, optimizer,
                latent, prompt_embeds, pooled_embeds, add_time_ids,
                device, dtype, args.grad_accum, step_idx,
            )
            scheduler_lr.step()
            total_loss += loss

            if step_idx % log_every == 0:
                lr_now = scheduler_lr.get_last_lr()[0]
                print(f"Ep {epoch+1:3d}/{args.epochs}  "
                      f"Step {step_idx:3d}/{len(loader)}  "
                      f"Loss {loss:.5f}  LR {lr_now:.2e}")

        avg = total_loss / len(loader)
        print(f"→ Epoch {epoch+1:3d} | 평균 손실: {avg:.5f}\n")

        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.output_dir,
                                f"pierr_probst_ep{epoch+1}.safetensors")
            save_as_lora(unet, ckpt, args.lora_rank)

    # ── 8. 최종 저장 ────────────────────────────────────────────────────────
    final = os.path.join(args.output_dir, "pierr_probst_sdctft_final.safetensors")
    print("\n[저장] 최종 LoRA 저장...")
    save_as_lora(unet, final, args.lora_rank)

    # ComfyUI loras 폴더에 복사
    import shutil
    comfy_lora = Path.home() / "ComfyUI" / "models" / "loras"
    if comfy_lora.exists():
        dst = comfy_lora / "pierr_probst_sdctft.safetensors"
        shutil.copy2(final, dst)
        print(f"[ComfyUI] LoRA 복사: {dst}")

    print(f"\n{'='*60}")
    print(f" 완료!")
    print(f" 베이스:  models/sd_xl_base_1.0.safetensors")
    print(f" LoRA :  {final}")
    print(f"")
    print(f" ComfyUI 사용:")
    print(f"   1. CheckpointLoaderSimple → sd_xl_base_1.0.safetensors")
    print(f"   2. LoraLoader → pierr_probst_sdctft.safetensors  (strength 0.7~1.0)")
    print(f"   3. 프롬프트에 'Pierr Probst style' 포함")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
