"""Real-ESRGAN (x4plus) upscaling for the clean card background.

The RRDBNet generator is implemented directly against the published weights so
no extra pip packages are needed (torch already comes with EasyOCR). Weights
are bundled with packaged builds; during development they are downloaded on
first use and cached next to this module.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "RealESRGAN_x4plus.pth"
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
NATIVE_SCALE = 4


# ---------------------------------------------------------------------------
# RRDBNet (Real-ESRGAN x4plus generator)
# ---------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# Weights (bundled or downloaded once)
# ---------------------------------------------------------------------------
def _model_path() -> Path:
    bundled = Path(__file__).resolve().parent / "models" / MODEL_NAME
    if bundled.exists() or not getattr(sys, "frozen", False):
        return bundled
    # Frozen single-file builds extract to a temporary directory, so a missing
    # model must go somewhere persistent.
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    return base / "PrimalCardRebuilder" / "models" / MODEL_NAME


def _download_weights(dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading Real-ESRGAN weights (~64 MB) to {dest} ...")
    with requests.get(MODEL_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0) or 0)
        written = 0
        last_pct = -10
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    pct = int(written * 100 / total)
                    if pct >= last_pct + 10:
                        print(f"  ... {pct}%")
                        last_pct = pct
    tmp.replace(dest)
    print("Real-ESRGAN weights ready.")
    return dest


def ensure_model_weights() -> Path:
    path = _model_path()
    if path.exists():
        return path
    return _download_weights(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class RealESRGANUpscaler:
    def __init__(self, tile=512, tile_pad=16):
        self.tile = tile
        self.tile_pad = tile_pad
        self.model = None
        self.device = None
        self.half = False
        self.last_engine = "real-esrgan"

    def _ensure_model(self):
        if self.model is not None:
            return
        weights = ensure_model_weights()
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        params = state.get("params_ema") or state.get("params") or state
        model = RRDBNet()
        model.load_state_dict(params, strict=True)
        model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.half = True
            model = model.half()
            print(f"Real-ESRGAN loaded on {torch.cuda.get_device_name(0)} (fp16).")
        else:
            self.device = torch.device("cpu")
            self.half = False
            print("Real-ESRGAN loaded on CPU (fp32).")
        self.model = model.to(self.device)

    def _to_tensor(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device).float().div_(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if self.half:
            tensor = tensor.half()
        return tensor

    @staticmethod
    def _to_bgr(output):
        out = output.squeeze(0).float().clamp_(0, 1).mul_(255.0).round_().byte()
        array = out.permute(1, 2, 0).contiguous().cpu().numpy()
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    def _infer(self, img_bgr):
        h, w = img_bgr.shape[:2]
        pad = max(0, min(self.tile_pad, h - 1, w - 1))
        if pad > 0:
            padded = cv2.copyMakeBorder(
                img_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
            )
        else:
            padded = img_bgr

        out = np.zeros((h * NATIVE_SCALE, w * NATIVE_SCALE, 3), dtype=np.uint8)
        with torch.inference_mode():
            for y in range(0, h, self.tile):
                for x in range(0, w, self.tile):
                    tw = min(self.tile, w - x)
                    th = min(self.tile, h - y)
                    # Include `pad` pixels of context around the core tile; the
                    # padded image supplies reflected borders at the edges.
                    crop = padded[y:y + th + 2 * pad, x:x + tw + 2 * pad]
                    result = self._to_bgr(self.model(self._to_tensor(crop)))
                    core = result[
                        pad * NATIVE_SCALE:(pad + th) * NATIVE_SCALE,
                        pad * NATIVE_SCALE:(pad + tw) * NATIVE_SCALE,
                    ]
                    out[
                        y * NATIVE_SCALE:(y + th) * NATIVE_SCALE,
                        x * NATIVE_SCALE:(x + tw) * NATIVE_SCALE,
                    ] = core
        return out

    def upscale(self, img_bgr, factor=4):
        """Upscale a BGR image by ``factor`` (1, 2 or 4). Never raises: if the
        model cannot be loaded, Lanczos is used instead and ``last_engine``
        reports the fallback."""
        if factor not in (1, 2, 4):
            raise ValueError(f"Unsupported upscale factor: {factor}")
        if factor == 1:
            self.last_engine = "none"
            return img_bgr

        try:
            self._ensure_model()
            result = self._infer(img_bgr)
            self.last_engine = "real-esrgan"
        except Exception as exc:
            print(f"Real-ESRGAN unavailable ({exc}); falling back to Lanczos.")
            h, w = img_bgr.shape[:2]
            result = cv2.resize(
                img_bgr,
                (w * NATIVE_SCALE, h * NATIVE_SCALE),
                interpolation=cv2.INTER_LANCZOS4,
            )
            self.last_engine = "lanczos-fallback"

        if factor == 2:
            h, w = img_bgr.shape[:2]
            result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        return result

    def upscale_file(self, input_path, output_path, factor=4):
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        result = self.upscale(img, factor)
        cv2.imwrite(str(output_path), result)
        return str(output_path)


_upscaler = None


def get_upscaler() -> RealESRGANUpscaler:
    global _upscaler
    if _upscaler is None:
        _upscaler = RealESRGANUpscaler()
    return _upscaler
