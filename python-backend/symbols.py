"""Golden symbol library and symbol detection/matching.

The runtime uses the transparent PNGs in ``symbols/`` as golden artwork.
Detection is layered:

* Template matching always supplies precise localisation and a ranked
  suggestion for every symbol in the library.
* Qwen3-VL supplies the primary identification (and rough boxes) by default.
  Every AI box is refined against the template library, so the final mask is
  always tightly fitted even when the model's coordinates are sloppy.
* Anything the AI misses can be recovered by a high-confidence template scan.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from llm_client import get_vision_model, vision_json

if getattr(sys, "frozen", False):
    SYMBOLS_DIR = Path(sys._MEIPASS) / "symbols"  # type: ignore[attr-defined]
else:
    SYMBOLS_DIR = Path(__file__).resolve().parent / "symbols"

# Detection runs on a width-normalised card so scale ranges are predictable.
REF_WIDTH = 320

# Template content height in pixels at REF_WIDTH.  These ranges cover the
# standard card layout (the three target symbols are 28-51 px here).
SIZE_RANGES = {
    "offensive": (26, 62),
    "defensive": (26, 64),
    "aggro_grey": (30, 68),
    "aggro_red": (30, 68),
    "stamina": (22, 42),
}
DEFAULT_SIZE_RANGE = (12, 90)

CORR_FLOOR = 0.86
SUGGEST_THRESHOLD = 0.58
REFINE_THRESHOLD = 0.66
AUTO_THRESHOLD = 0.75
FALLBACK_THRESHOLD = 0.78
MAX_PEAKS_PER_SCALE = 10

_DARK_THRESHOLD = 30.0
_BRIGHT_THRESHOLD = 30.0


@dataclass
class Candidate:
    symbol_id: str
    name: str
    x: int
    y: int
    w: int
    h: int
    corr: float
    iou: float
    contrast: float
    color: float
    score: float
    alternates: list["Candidate"] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class _Template:
    id: str
    name: str
    category: str
    description: str
    vision_description: str
    file: str
    svg: Optional[str]
    color: str
    polarity: str
    path: Path
    gray: np.ndarray
    bgr: np.ndarray
    mask_u8: np.ndarray
    color_mask_u8: np.ndarray
    lab_color: np.ndarray
    _scale_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)

    def scaled(self, target_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._scale_cache.get(target_h)
        if cached is not None:
            return cached
        scale = target_h / self.gray.shape[0]
        target_w = max(3, int(round(self.gray.shape[1] * scale)))
        gray = cv2.resize(self.gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        color_mask = cv2.resize(
            self.color_mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        result = (gray, mask, color_mask)
        self._scale_cache[target_h] = result
        return result


class SymbolLibrary:
    def __init__(self, directory: Path = SYMBOLS_DIR):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.templates: list[_Template] = []
        self.by_id: dict[str, _Template] = {}
        self._pil_cache: dict[str, Image.Image] = {}
        for entry in self.manifest.get("symbols", []):
            template = self._load_template(entry)
            self.templates.append(template)
            self.by_id[template.id] = template

    def _load_template(self, entry: dict) -> _Template:
        path = self.directory / entry["file"]
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Could not read symbol asset: {path}")
        if bgra.shape[2] == 3:
            bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
        alpha = bgra[:, :, 3]
        mask_u8 = ((alpha > 24) * 255).astype(np.uint8)
        bbox = cv2.boundingRect(mask_u8)
        x, y, w, h = bbox
        bgra = bgra[y : y + h, x : x + w]
        mask_u8 = mask_u8[y : y + h, x : x + w]
        bgr = bgra[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # The colour check ignores the dark diamond behind the Aggro head and
        # keeps the bright head/glyph pixels where the colour is identifiable.
        color_mask_u8 = ((mask_u8 > 0) & (gray > 80)).astype(np.uint8) * 255
        if not np.any(color_mask_u8):
            color_mask_u8 = mask_u8.copy()
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_color = lab[color_mask_u8 > 0].mean(axis=0)
        return _Template(
            id=entry["id"],
            name=entry["name"],
            category=entry.get("category", ""),
            description=entry.get("description", ""),
            vision_description=entry.get("vision_description") or entry.get("description", ""),
            file=entry["file"],
            svg=entry.get("svg"),
            color=entry.get("color", ""),
            polarity=entry.get("polarity", "bright"),
            path=path,
            gray=gray,
            bgr=bgr,
            mask_u8=mask_u8,
            color_mask_u8=color_mask_u8,
            lab_color=lab_color,
        )

    # -- public metadata ---------------------------------------------------
    def public_symbols(self) -> list[dict]:
        return [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "color": t.color,
                "polarity": t.polarity,
                "url": f"/api/symbols/asset/{t.id}",
            }
            for t in self.templates
        ]

    def asset_path(self, symbol_id: str) -> Optional[Path]:
        template = self.by_id.get(symbol_id)
        return template.path if template else None

    def pil_asset(self, symbol_id: str) -> Optional[Image.Image]:
        """Golden artwork cropped to its visible pixels, for compositing."""
        if symbol_id in self._pil_cache:
            return self._pil_cache[symbol_id]
        template = self.by_id.get(symbol_id)
        if template is None:
            return None
        image = Image.open(template.path).convert("RGBA")
        bbox = image.getchannel("A").getbbox()
        if bbox:
            image = image.crop(bbox)
        self._pil_cache[symbol_id] = image
        return image

    # -- template matching -------------------------------------------------
    def _size_range(self, symbol_id: str) -> tuple[int, int]:
        return SIZE_RANGES.get(symbol_id, DEFAULT_SIZE_RANGE)

    @staticmethod
    def _prepare_image(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = bgr.shape[:2]
        ref_scale = REF_WIDTH / float(width)
        if abs(ref_scale - 1.0) > 1e-6:
            interpolation = cv2.INTER_AREA if ref_scale < 1.0 else cv2.INTER_CUBIC
            ref = cv2.resize(
                bgr,
                (REF_WIDTH, max(1, int(round(height * ref_scale)))),
                interpolation=interpolation,
            )
        else:
            ref = bgr
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
        return ref, gray, lab, ref_scale

    def _verify(
        self,
        gray: np.ndarray,
        lab: np.ndarray,
        px: int,
        py: int,
        template: _Template,
        scaled_gray: np.ndarray,
        scaled_mask: np.ndarray,
        scaled_color_mask: np.ndarray,
        corr: float,
    ) -> Optional[Candidate]:
        h, w = scaled_gray.shape
        patch = gray[py : py + h, px : px + w]
        if patch.shape[:2] != (h, w):
            return None
        mask = scaled_mask > 0
        if not np.any(mask):
            return None

        ring_size = max(3, int(round(min(w, h) * 0.22)) | 1)
        ring = (
            cv2.dilate(scaled_mask, np.ones((ring_size, ring_size), np.uint8)) > 0
        ) & ~mask
        if np.any(ring):
            background = float(np.median(patch[ring]))
        elif np.any(~mask):
            background = float(np.median(patch[~mask]))
        else:
            background = 0.0

        if template.polarity == "bright":
            foreground = patch > background + _BRIGHT_THRESHOLD
            contrast = float(np.median(patch[mask]) - background)
            if contrast < 20.0:
                return None
        else:
            foreground = patch < background - _DARK_THRESHOLD
            contrast = float(np.median(patch[mask]) - background)
            if contrast > -20.0:
                return None

        intersection = float(np.count_nonzero(foreground & mask))
        union = float(np.count_nonzero(foreground | mask))
        iou = intersection / union if union else 0.0
        if iou < 0.18:
            return None

        color_mask = scaled_color_mask > 0
        if np.any(color_mask):
            patch_lab = lab[py : py + h, px : px + w][color_mask].mean(axis=0)
            delta_e = float(np.linalg.norm(patch_lab - template.lab_color))
            color_score = math.exp(-delta_e / 22.0)
        else:
            color_score = 0.5

        score = 0.50 * corr + 0.38 * iou + 0.12 * color_score
        return Candidate(
            symbol_id=template.id,
            name=template.name,
            x=int(px),
            y=int(py),
            w=int(w),
            h=int(h),
            corr=float(corr),
            iou=float(iou),
            contrast=float(contrast),
            color=float(color_score),
            score=float(score),
        )

    def _search(
        self,
        gray: np.ndarray,
        lab: np.ndarray,
        symbols: Optional[list[str]] = None,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> list[Candidate]:
        image_h, image_w = gray.shape[:2]
        if region is None:
            x0, y0, x1, y1 = 0, 0, image_w, image_h
        else:
            x0, y0, x1, y1 = region
            x0 = max(0, min(image_w, int(x0)))
            y0 = max(0, min(image_h, int(y0)))
            x1 = max(x0 + 1, min(image_w, int(x1)))
            y1 = max(y0 + 1, min(image_h, int(y1)))
        crop_gray = gray[y0:y1, x0:x1]
        crop_lab = lab[y0:y1, x0:x1]
        crop_h, crop_w = crop_gray.shape[:2]

        selected = self.templates
        if symbols is not None:
            wanted = set(symbols)
            selected = [t for t in self.templates if t.id in wanted]

        candidates: list[Candidate] = []
        for template in selected:
            low, high = self._size_range(template.id)
            low = max(8, low)
            high = min(high, crop_h, int(crop_w * template.gray.shape[0] / template.gray.shape[1]) + 2)
            if high <= low:
                continue
            for target_h in range(low, high + 1, 2):
                scaled_gray, scaled_mask, scaled_color_mask = template.scaled(target_h)
                target_w = scaled_gray.shape[1]
                if target_w > crop_w or target_h > crop_h:
                    continue
                result = cv2.matchTemplate(
                    crop_gray, scaled_gray, cv2.TM_CCORR_NORMED, mask=scaled_mask
                )
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                dilated = cv2.dilate(result, np.ones((5, 5), np.uint8))
                ys, xs = np.where((result >= dilated - 1e-6) & (result >= CORR_FLOOR))
                if len(ys) == 0:
                    continue
                if len(ys) > MAX_PEAKS_PER_SCALE:
                    order = np.argsort(result[ys, xs])[-MAX_PEAKS_PER_SCALE:]
                    ys, xs = ys[order], xs[order]
                for py, px in zip(ys.tolist(), xs.tolist()):
                    corr = float(result[py, px])
                    candidate = self._verify(
                        crop_gray,
                        crop_lab,
                        px,
                        py,
                        template,
                        scaled_gray,
                        scaled_mask,
                        scaled_color_mask,
                        corr,
                    )
                    if candidate is None:
                        continue
                    candidate.x += x0
                    candidate.y += y0
                    if candidate.score >= SUGGEST_THRESHOLD:
                        candidates.append(candidate)
        return candidates

    # -- public detection --------------------------------------------------
    def detect_template(self, image_path: str) -> tuple[list[dict], list[str]]:
        _ref, gray, lab, ref_scale = self._prepare_image(image_path)
        candidates = self._search(gray, lab)
        accepted = _non_max_suppression(candidates)
        accepted = [c for c in accepted if c.score >= AUTO_THRESHOLD]
        # A card normally has at most one of each front-facing symbol.
        seen_ids: set[str] = set()
        rows = []
        for candidate in accepted:
            if candidate.symbol_id in seen_ids:
                continue
            seen_ids.add(candidate.symbol_id)
            rows.append(self._candidate_to_row(candidate, ref_scale, method="template"))
        rows.sort(key=lambda row: (row["bbox"]["y_min"], row["bbox"]["x_min"]))
        warnings = [] if rows else ["Template matching found no symbols on this card."]
        return rows, warnings

    def detect_ai(self, image_path: str) -> tuple[list[dict], list[str], str]:
        """AI-first detection with template localisation and recovery."""
        warnings: list[str] = []
        vision_model = get_vision_model()
        if not vision_model:
            rows, template_warnings = self.detect_template(image_path)
            return (
                rows,
                ["No vision model is loaded; used template matching instead."]
                + template_warnings,
                "",
            )

        ref, gray, lab, ref_scale = self._prepare_image(image_path)
        original_width, original_height = _image_size(image_path)
        catalog = [
            {
                "id": t.id,
                "name": t.name,
                "description": t.vision_description,
            }
            for t in self.templates
        ]
        system_prompt = (
            "You are a precise symbol detector for the Primal: The Awakening card game. "
            "You inspect one card image and locate symbols from a fixed catalog. "
            "Reply with JSON only, no markdown. All bounding-box coordinates MUST be normalized "
            "floats between 0.0 and 1.0 relative to image width and height, where (0,0) is the "
            "top-left corner and (1,1) is the bottom-right corner. Bounding boxes must tightly "
            "enclose each symbol. Only report catalog symbols that are actually visible."
        )
        user_prompt = (
            "Catalog:\n"
            + json.dumps(catalog, indent=1, ensure_ascii=False)
            + "\n\nImportant: the diamond at the TOP-LEFT containing a number is the Stamina "
            "COST, not an Aggro symbol and not a Defensive symbol. Never report it. The Aggro "
            "symbol is only the dark diamond with a horned beast head at the top-right. The "
            "tiny white Stamina drop is at the bottom-left near the bottom band; look carefully "
            "and report it.\n\n"
            "Return JSON exactly like:\n"
            '{"symbols":[{"id":"one catalog id","bbox":[x1,y1,x2,y2],"confidence":0.0}]}\n'
            "bbox values are normalized 0..1. confidence is 0..1. JSON only."
        )
        data, meta = vision_json(
            image_path, system_prompt, user_prompt, model=vision_model, max_tokens=600
        )
        if data is None:
            rows, template_warnings = self.detect_template(image_path)
            return (
                rows,
                [f"AI detection failed ({meta.get('error', 'unknown error')}); used template matching."]
                + template_warnings,
                "",
            )

        entries = _extract_ai_entries(data)
        if not entries:
            rows, template_warnings = self.detect_template(image_path)
            return (
                rows,
                ["AI returned no usable symbol boxes; used template matching."]
                + template_warnings,
                vision_model,
            )

        known_ids = set(self.by_id)
        ai_rows: list[dict] = []
        for entry in entries:
            symbol_id = str(entry.get("id") or entry.get("symbol_id") or "").strip()
            if symbol_id not in known_ids:
                if symbol_id:
                    warnings.append(f"Ignored AI detection with unknown symbol id '{symbol_id}'.")
                continue
            bbox_original = _entry_bbox(entry, original_width, original_height, meta)
            if bbox_original is None:
                continue
            suggestions = self.suggest_for_box(image_path, bbox_original)
            best = suggestions[0] if suggestions else None
            alternates = [
                suggestion
                for suggestion in suggestions
                if suggestion["symbol_id"] != symbol_id
            ]
            bbox = bbox_original
            template_id = None
            template_score = None
            if best is not None:
                template_id = best["symbol_id"]
                template_score = float(best["score"])
                if best["score"] >= REFINE_THRESHOLD:
                    bbox = best["bbox"]
            confidence = _clamp_float(entry.get("confidence"), 0.0, 1.0, 0.5)
            warning = ""
            if best is None or best["score"] < SUGGEST_THRESHOLD:
                warning = "No confident template match; box is from AI."
            elif template_id != symbol_id:
                warning = (
                    f"AI matched {self.by_id[symbol_id].name}, "
                    f"template suggests {self.by_id[template_id].name} "
                    f"({template_score:.2f})."
                )
            ai_rows.append(
                {
                    "row_id": _row_id("auto"),
                    "symbol_id": symbol_id,
                    "bbox": bbox,
                    "source": "auto",
                    "method": "ai",
                    "confidence": confidence,
                    "template_id": template_id,
                    "template_score": template_score,
                    "alternates": alternates,
                    "warning": warning,
                    "redraw": True,
                    "inpaint": True,
                }
            )

        # Primary AI rows may collapse onto the same symbol (duplicates); keep
        # the highest-confidence box for each id.
        ai_rows.sort(key=lambda row: -row["confidence"])
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in ai_rows:
            if row["symbol_id"] in seen:
                continue
            seen.add(row["symbol_id"])
            deduped.append(row)
        ai_rows = deduped

        # Recover symbols the model missed with a strict template-only scan.
        missing = [t.id for t in self.templates if t.id not in seen]
        if missing:
            missing_candidates = [
                c
                for c in self._search(gray, lab, symbols=missing)
                if c.score >= FALLBACK_THRESHOLD
            ]
            accepted = _non_max_suppression(missing_candidates)
            for candidate in accepted:
                candidate_bbox = _bbox_tuple(_scale_bbox(candidate, ref_scale))
                if any(
                    _bbox_iou(candidate_bbox, _bbox_tuple(row["bbox"])) > 0.25
                    for row in ai_rows
                ):
                    continue
                row = self._candidate_to_row(candidate, ref_scale, method="template")
                row["warning"] = "AI did not report this symbol; found by template matching."
                ai_rows.append(row)

        ai_rows.sort(key=lambda row: (row["bbox"]["y_min"], row["bbox"]["x_min"]))
        if not ai_rows:
            rows, template_warnings = self.detect_template(image_path)
            return (
                rows,
                ["AI returned no usable symbol boxes; used template matching."]
                + template_warnings,
                vision_model,
            )
        if not seen:
            warnings.append("AI did not identify any catalog symbols.")
        return ai_rows, warnings, vision_model

    def suggest_for_box(
        self, image_path: str, bbox: dict, top_k: int = 4
    ) -> list[dict]:
        """Return ranked golden-symbol suggestions for a single bounding box."""
        _ref, gray, lab, ref_scale = self._prepare_image(image_path)
        x1 = float(bbox["x_min"]) * ref_scale
        y1 = float(bbox["y_min"]) * ref_scale
        x2 = float(bbox["x_max"]) * ref_scale
        y2 = float(bbox["y_max"]) * ref_scale
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 2 or box_h < 2:
            return []
        pad = int(round(max(box_w, box_h) * 0.6)) + 6
        image_h, image_w = gray.shape[:2]
        region = (
            max(0, int(math.floor(x1 - pad))),
            max(0, int(math.floor(y1 - pad))),
            min(image_w, int(math.ceil(x2 + pad))),
            min(image_h, int(math.ceil(y2 + pad))),
        )
        candidates = self._search(gray, lab, region=region)
        best_by_id: dict[str, Candidate] = {}
        for candidate in sorted(candidates, key=lambda c: -c.score):
            if candidate.symbol_id not in best_by_id:
                best_by_id[candidate.symbol_id] = candidate
        ranked = sorted(best_by_id.values(), key=lambda c: -c.score)[:top_k]
        return [
            {
                "symbol_id": candidate.symbol_id,
                "name": candidate.name,
                "score": round(candidate.score, 4),
                "bbox": _scale_bbox(candidate, ref_scale),
            }
            for candidate in ranked
        ]

    @staticmethod
    def _candidate_to_row(candidate: Candidate, ref_scale: float, method: str) -> dict:
        alternates = [
            {
                "symbol_id": alt.symbol_id,
                "name": alt.name,
                "score": round(alt.score, 4),
                "bbox": _scale_bbox(alt, ref_scale),
            }
            for alt in candidate.alternates[:3]
        ]
        return {
            "row_id": _row_id("auto"),
            "symbol_id": candidate.symbol_id,
            "bbox": _scale_bbox(candidate, ref_scale),
            "source": "auto",
            "method": method,
            "confidence": round(candidate.score, 4),
            "template_id": candidate.symbol_id,
            "template_score": round(candidate.score, 4),
            "alternates": alternates,
            "warning": "",
            "redraw": True,
            "inpaint": True,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _row_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _extract_ai_entries(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        for key in ("symbols", "detections", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if "id" in data:
            return [data]
    return []


def _entry_bbox(
    entry: dict, width: int, height: int, meta: dict
) -> Optional[dict]:
    raw = entry.get("bbox") or entry.get("box")
    if isinstance(raw, dict):
        try:
            values = [
                float(raw["x_min"]),
                float(raw["y_min"]),
                float(raw["x_max"]),
                float(raw["y_max"]),
            ]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            values = [float(value) for value in raw[:4]]
        except (TypeError, ValueError):
            return None
    else:
        return None

    if max(values) <= 1.5:
        x1, y1, x2, y2 = values
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    else:
        sent_w = meta.get("width") or width
        sent_h = meta.get("height") or height
        x1, y1, x2, y2 = values
        x1 = x1 / sent_w * width
        x2 = x2 / sent_w * width
        y1 = y1 / sent_h * height
        y2 = y2 / sent_h * height

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return {
        "x_min": int(round(min(x1, x2))),
        "y_min": int(round(min(y1, y2))),
        "x_max": int(round(max(x1, x2))),
        "y_max": int(round(max(y1, y2))),
    }


def _scale_bbox(candidate: Candidate, ref_scale: float) -> dict:
    x1 = candidate.x / ref_scale
    y1 = candidate.y / ref_scale
    x2 = (candidate.x + candidate.w) / ref_scale
    y2 = (candidate.y + candidate.h) / ref_scale
    return {
        "x_min": int(round(x1)),
        "y_min": int(round(y1)),
        "x_max": int(round(x2)),
        "y_max": int(round(y2)),
    }


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _bbox_iou_dict(a: dict, b: dict) -> float:
    return _bbox_iou(_bbox_tuple(a), _bbox_tuple(b))


def _bbox_tuple(box: dict) -> tuple[int, int, int, int]:
    return (box["x_min"], box["y_min"], box["x_max"], box["y_max"])


def _non_max_suppression(candidates: list[Candidate], iou_threshold: float = 0.35) -> list[Candidate]:
    accepted: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: -c.score):
        overlap = None
        for existing in accepted:
            if _bbox_iou(candidate.bbox, existing.bbox) > iou_threshold:
                overlap = existing
                break
        if overlap is None:
            accepted.append(candidate)
        elif (
            candidate.symbol_id != overlap.symbol_id
            and len(overlap.alternates) < 3
            and all(alt.symbol_id != candidate.symbol_id for alt in overlap.alternates)
        ):
            overlap.alternates.append(candidate)
    return accepted


_library: Optional[SymbolLibrary] = None


def get_symbol_library() -> SymbolLibrary:
    global _library
    if _library is None:
        _library = SymbolLibrary()
    return _library


def detect_symbols(image_path: str, mode: str = "ai") -> dict:
    library = get_symbol_library()
    mode = (mode or "ai").lower()
    if mode == "template":
        rows, warnings = library.detect_template(image_path)
        return {
            "symbols": rows,
            "mode": "template",
            "vision_model": "",
            "warnings": warnings,
        }
    rows, warnings, vision_model = library.detect_ai(image_path)
    return {
        "symbols": rows,
        "mode": "ai" if vision_model else "template",
        "vision_model": vision_model,
        "warnings": warnings,
    }
