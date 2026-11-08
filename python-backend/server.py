import argparse
import shutil
import sys

# The Rust shell pipes this process's stdout/stderr and decodes them as UTF-8.
# On Windows, Python otherwise uses the locale code page (cp1252) for pipes,
# which raises UnicodeEncodeError as soon as a log line contains non-ASCII.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from backend_core import CardProcessor
from llm_client import fix_ocr_text, get_loaded_model
from symbols import detect_symbols, get_symbol_library
from upscaler import get_upscaler

app = FastAPI(title="Primal Card Rebuilder")

# The Tauri webview runs on a different origin (http://tauri.localhost on
# Windows) than this sidecar, so every fetch from the UI is cross-origin.
# Without this middleware the browser blocks all API calls (including the
# preflight for the JSON POST to /api/load).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy OCR processor: instantiated on first OCR call, never at import time.
processor: Optional[CardProcessor] = None

# Runtime state
current_image_path: Optional[str] = None
ocr_results: list = []
manual_mask: list = []
symbol_results: list = []

if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)
    OUTPUT_DIR = _BASE_DIR / "output"
else:
    _BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = _BASE_DIR.parent / "output"

FONTS_DIR = _BASE_DIR / "fonts"
DATA_DIR = _BASE_DIR / "data"


def get_processor() -> CardProcessor:
    global processor
    if processor is None:
        processor = CardProcessor()
    return processor


class LoadRequest(BaseModel):
    path: str


class PipelineStateRequest(BaseModel):
    # OCR rows plus manually drawn rectangle additions. The UI is the source
    # of truth for both, so edits (text, inpaint/redraw flags, extra boxes)
    # are honoured without needing separate sync calls.
    results: Optional[list] = None
    extra_mask: Optional[list] = None
    # Symbol rows: detected/manual boxes with a golden symbol assignment.
    symbols: Optional[list] = None
    # 1 = no upscale, 2 or 4 = Real-ESRGAN output scale.
    upscale: int = 4


class SymbolDetectRequest(BaseModel):
    # "ai" = AI-first with template localisation/backup, "template" = CV only.
    mode: str = "ai"


class SymbolMatchRequest(BaseModel):
    bbox: dict


def _resolve_pipeline_state(req: Optional[PipelineStateRequest]):
    results = req.results if req is not None and req.results is not None else ocr_results
    extra_mask = req.extra_mask if req is not None and req.extra_mask is not None else manual_mask
    symbols = req.symbols if req is not None and req.symbols is not None else symbol_results
    return results, extra_mask, symbols


def _resolve_upscale(req: Optional[PipelineStateRequest]) -> int:
    value = req.upscale if req is not None else 4
    return value if value in (1, 2, 4) else 4


def _symbol_warnings(symbols) -> list:
    unassigned = sum(
        1
        for item in symbols or []
        if item.get("redraw", True) and not item.get("symbol_id")
    )
    if unassigned:
        return [
            f"{unassigned} symbol box(es) have no golden symbol assigned and were left untouched."
        ]
    return []


def _split_lines(corrected_text):
    if not corrected_text:
        return []
    return [line.strip() for line in corrected_text.split("\n") if line.strip()]


@app.get("/api/health")
def health():
    """Cheap readiness probe; does not touch OCR or the LLM."""
    return {"ok": True}


@app.get("/api/model")
def get_model():
    return {"model": get_loaded_model()}


@app.post("/api/load")
def load_image(req: LoadRequest):
    global current_image_path, ocr_results, manual_mask, symbol_results
    src = Path(req.path)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / "original.png"
    if src.resolve() != dest.resolve():
        shutil.copyfile(str(src), str(dest))
    current_image_path = str(src)
    # A new card invalidates any OCR/symbol/rendered state from the previous one.
    ocr_results = []
    manual_mask = []
    symbol_results = []
    for stale in ("clean.png", "final.png", "clean_bg.png", "final_card.png", "clean_upscaled.png"):
        (OUTPUT_DIR / stale).unlink(missing_ok=True)
    return {"url": "/api/image/original"}


@app.post("/api/ocr")
def run_ocr():
    global ocr_results
    if not current_image_path:
        raise HTTPException(status_code=400, detail="No image loaded.")
    proc = get_processor()
    results = proc.extract_text(str(current_image_path), str(OUTPUT_DIR / "ocr_data.json"))
    ocr_results = []
    for item in results:
        ocr_results.append({
            "text": item.get("text", ""),
            "confidence": item.get("confidence", 0.0),
            "bbox": item.get("bbox", {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}),
            "ai_modified": False,
            "inpaint": True,
            "redraw": True,
        })
    return ocr_results


@app.post("/api/ai-fix")
def ai_fix(req: Optional[PipelineStateRequest] = None):
    global ocr_results
    results, _extra_mask, _symbols = _resolve_pipeline_state(req)
    if not results:
        raise HTTPException(status_code=400, detail="No OCR results. Run OCR first.")
    ocr_results = results
    fragments = [item.get("text", "") for item in ocr_results]
    result = fix_ocr_text(fragments, return_context=True)
    if isinstance(result, tuple):
        corrected_text, context = result
    else:
        corrected_text, context = result, ""
    corrected_lines = _split_lines(corrected_text)

    # Merge corrected lines back into ocr_results; never blank rows the LLM omitted.
    for i, item in enumerate(ocr_results):
        if i < len(corrected_lines):
            new_text = corrected_lines[i]
            item["ai_modified"] = new_text != item.get("text", "")
            item["text"] = new_text
        else:
            item["ai_modified"] = False

    # Overflow: append extra AI lines onto the last fragment
    if len(corrected_lines) > len(ocr_results) and ocr_results:
        extra_text = " ".join(corrected_lines[len(ocr_results):])
        ocr_results[-1]["text"] += " " + extra_text
        ocr_results[-1]["ai_modified"] = True

    return {"corrected": corrected_lines, "context": context or ""}


@app.post("/api/inpaint")
def inpaint_only(req: Optional[PipelineStateRequest] = None):
    """Regenerate only the clean background from the current mask state."""
    global ocr_results, manual_mask, symbol_results
    if not current_image_path:
        raise HTTPException(status_code=400, detail="No image loaded.")
    results, extra_mask, symbols = _resolve_pipeline_state(req)
    ocr_results = results
    manual_mask = extra_mask
    symbol_results = symbols
    proc = get_processor()
    proc.inpaint_card(
        str(current_image_path),
        results,
        str(OUTPUT_DIR / "clean.png"),
        extra_boxes=extra_mask,
        symbols=symbols,
    )
    return {"ok": True, "warnings": _symbol_warnings(symbols)}


@app.post("/api/render")
def render_card(req: Optional[PipelineStateRequest] = None):
    global ocr_results, manual_mask, symbol_results
    if not current_image_path:
        raise HTTPException(status_code=400, detail="No card loaded.")
    results, extra_mask, symbols = _resolve_pipeline_state(req)
    drawable_symbols = [
        item
        for item in symbols or []
        if item.get("symbol_id") and item.get("redraw", True) is not False
    ]
    if not results and not extra_mask and not drawable_symbols:
        raise HTTPException(
            status_code=400,
            detail="No OCR results, mask regions, or assigned symbols to render.",
        )
    upscale = _resolve_upscale(req)
    ocr_results = results
    manual_mask = extra_mask
    symbol_results = symbols
    proc = get_processor()
    # These names must match what /api/images and /api/image/{name} serve.
    clean_bg = proc.inpaint_card(
        str(current_image_path),
        results,
        str(OUTPUT_DIR / "clean.png"),
        extra_boxes=extra_mask,
        symbols=symbols,
    )

    background = clean_bg
    engine = "none"
    if upscale > 1:
        upscaler = get_upscaler()
        background = upscaler.upscale_file(
            clean_bg, str(OUTPUT_DIR / "clean_upscaled.png"), upscale
        )
        engine = upscaler.last_engine
    else:
        (OUTPUT_DIR / "clean_upscaled.png").unlink(missing_ok=True)

    final_path = str(OUTPUT_DIR / "final.png")
    proc.render_text(background, results, final_path, coord_scale=upscale)
    proc.render_symbols(final_path, symbols, output_path=final_path, coord_scale=upscale)
    with Image.open(final_path) as final_img:
        size = list(final_img.size)
    return {
        "ok": True,
        "upscale": upscale,
        "engine": engine,
        "size": size,
        "symbols_drawn": len(drawable_symbols),
        "warnings": _symbol_warnings(symbols),
    }


@app.get("/api/symbols/library")
def symbols_library():
    """Golden symbol catalog, including runtime asset URLs."""
    library = get_symbol_library()
    return {"symbols": library.public_symbols()}


@app.get("/api/symbols/asset/{symbol_id}")
def symbol_asset(symbol_id: str):
    library = get_symbol_library()
    path = library.asset_path(symbol_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown symbol asset: {symbol_id}")
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"}
    )


@app.post("/api/symbols/detect")
def symbols_detect(req: SymbolDetectRequest):
    global symbol_results
    if not current_image_path:
        raise HTTPException(status_code=400, detail="No image loaded.")
    mode = (req.mode or "ai").lower()
    if mode not in ("ai", "template"):
        raise HTTPException(status_code=400, detail="mode must be 'ai' or 'template'.")
    try:
        result = detect_symbols(current_image_path, mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol detection failed: {e}")
    symbol_results = result.get("symbols", [])
    return result


@app.post("/api/symbols/match")
def symbols_match(req: SymbolMatchRequest):
    if not current_image_path:
        raise HTTPException(status_code=400, detail="No image loaded.")
    try:
        suggestions = get_symbol_library().suggest_for_box(
            current_image_path, req.bbox
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol matching failed: {e}")
    return {"suggestions": suggestions}


@app.get("/api/images")
def list_images():
    urls = {}
    for name in ("original", "clean", "final"):
        p = OUTPUT_DIR / f"{name}.png"
        if p.exists():
            urls[name] = f"/api/image/{name}"
    return urls


@app.get("/api/image/{name}")
def get_image(name: str):
    if name not in ("original", "clean", "final"):
        raise HTTPException(status_code=404, detail="Unknown image name")
    path = OUTPUT_DIR / f"{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {name}")
    return FileResponse(path, media_type="image/png")


def main():
    parser = argparse.ArgumentParser(description="Primal Card Rebuilder FastAPI sidecar")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
