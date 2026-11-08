#!/usr/bin/env python3
"""Extract golden symbol artwork from the Primal rulebook PDF.

The rulebook draws the small action-card symbols as vector paths on pages 25,
28 and 48 of the PDF (printed pages 24, 25 and 41).  This one-off developer
tool selects the relevant drawing objects, rebuilds them as SVG paths, and
rasterizes high-resolution transparent PNGs for the runtime symbol library.

Usage (from the repository root):
    venv\\Scripts\\python.exe scripts\\extract-rulebook-symbols.py

The output is written to python-backend/symbols/ plus a visual contact sheet at
output/symbols_contact_sheet.png.  Runtime code only needs the generated PNGs,
the SVGs and manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "python-backend" / "data" / "rulebook.pdf"
OUT_DIR = ROOT / "python-backend" / "symbols"
SHEET_PATH = ROOT / "output" / "symbols_contact_sheet.png"

# Long side of every generated raster.  The card's largest symbol is about
# 30 px on a 240 px-wide card, so this is >10x the 4x render target.
TARGET_PX = 512
PAD_PT = 0.5

# Drawing indices are stable for this exact PDF revision.  Each entry lists the
# drawing objects in paint order; ``color`` overrides the PDF fill (used to
# recolor the dark rulebook drop white for the card artwork).
ASSET_SPECS = [
    {
        "file": "offensive_white",
        "page": 27,
        "drawings": [(365, (1.0, 1.0, 1.0))],
    },
    {
        "file": "defensive_white",
        "page": 27,
        "drawings": [(378, (1.0, 1.0, 1.0)), (379, (1.0, 1.0, 1.0))],
    },
    {
        "file": "aggro_grey",
        "page": 26,
        "drawings": [
            (700, (0.1477, 0.1619, 0.1729)),
            (701, (0.1477, 0.1619, 0.1729)),
            (702, (0.49, 0.49, 0.49)),
        ],
    },
    {
        "file": "aggro_red",
        "page": 26,
        "drawings": [
            (697, (0.1477, 0.1619, 0.1729)),
            (698, (0.1477, 0.1619, 0.1729)),
            (699, (0.6374, 0.3102, 0.3187)),
        ],
    },
    {
        "file": "stamina_white",
        "page": 46,
        "drawings": [(697, (1.0, 1.0, 1.0))],
    },
]

SYMBOLS = [
    {
        "id": "offensive",
        "name": "Offensive",
        "category": "stance",
        "file": "offensive_white.png",
        "svg": "offensive_white.svg",
        "color": "white",
        "polarity": "bright",
        "description": (
            "White Offensive glyph: a stylized crown or anvil with a vertical "
            "spike, a horizontal crossbar, and an inverted triangle below it. "
            "Appears at the top-center of Offensive action cards."
        ),
        "vision_description": (
            "White crown/anvil glyph with a vertical spike, wide crossbar and "
            "inverted triangle below; located at the top-center of the card."
        ),
        "keywords": ["offensive", "offense", "crown", "anvil", "maneuver", "attack"],
    },
    {
        "id": "defensive",
        "name": "Defensive",
        "category": "stance",
        "file": "defensive_white.png",
        "svg": "defensive_white.svg",
        "color": "white",
        "polarity": "bright",
        "description": (
            "White Defensive glyph: a spiked diamond outline with a hollow "
            "diamond center and short thorns radiating outward. Appears at the "
            "top-center of Defensive action cards."
        ),
        "vision_description": (
            "White spiked diamond outline with a hollow diamond center and "
            "short radiating thorns; located at the top-center of the card."
        ),
        "keywords": ["defensive", "defense", "parry", "dodge", "spiked diamond"],
    },
    {
        "id": "aggro_grey",
        "name": "Aggro (grey)",
        "category": "aggro",
        "file": "aggro_grey.png",
        "svg": "aggro_grey.svg",
        "color": "grey",
        "polarity": "dark",
        "description": (
            "Aggro symbol, grey variant: a dark diamond with torn edges "
            "containing a light grey horned beast head. Marks a card that is "
            "not an Aggro card. Appears at the top-right of action cards."
        ),
        "vision_description": (
            "Dark torn-edge diamond containing a light grey horned beast head; "
            "located at the top-right corner of the card."
        ),
        "keywords": ["aggro", "grey", "gray", "beast head", "diamond"],
    },
    {
        "id": "aggro_red",
        "name": "Aggro (red)",
        "category": "aggro",
        "file": "aggro_red.png",
        "svg": "aggro_red.svg",
        "color": "red",
        "polarity": "dark",
        "description": (
            "Aggro symbol, red variant: a dark diamond with torn edges "
            "containing a muted red horned beast head. Marks an Aggro card. "
            "Appears at the top-right of action cards."
        ),
        "vision_description": (
            "Dark torn-edge diamond containing a red horned beast head; "
            "located at the top-right corner of the card."
        ),
        "keywords": ["aggro", "red", "beast head", "diamond"],
    },
    {
        "id": "stamina",
        "name": "Stamina",
        "category": "resource",
        "file": "stamina_white.png",
        "svg": "stamina_white.svg",
        "color": "white",
        "polarity": "bright",
        "description": (
            "White Stamina drop: a teardrop shape with a small crescent notch "
            "on the left side. Appears at the bottom-left of action cards next "
            "to the Stamina cost."
        ),
        "vision_description": (
            "Small white teardrop/drop with a notch on the left; located at the "
            "bottom-left corner near the bottom band."
        ),
        "keywords": ["stamina", "drop", "teardrop", "cost", "resource"],
    },
]


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _hex_color(rgb) -> str:
    return "#" + "".join(
        f"{max(0, min(255, round(channel * 255))):02x}" for channel in rgb
    )


def _drawing_to_svg_path(drawing: dict) -> str:
    """Convert a PyMuPDF drawing's path items into SVG path commands.

    All subpaths stay in a single ``<path>`` element so nonzero winding (and
    therefore holes) is preserved exactly as in the PDF.
    """
    parts: list[str] = []
    current = None
    for item in drawing["items"]:
        kind = item[0]
        if kind == "l":
            p1, p2 = item[1], item[2]
            if current is None or abs(p1.x - current.x) > 1e-4 or abs(p1.y - current.y) > 1e-4:
                parts.append(f"M{_fmt(p1.x)} {_fmt(p1.y)}")
            parts.append(f"L{_fmt(p2.x)} {_fmt(p2.y)}")
            current = p2
        elif kind == "c":
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            if current is None or abs(p1.x - current.x) > 1e-4 or abs(p1.y - current.y) > 1e-4:
                parts.append(f"M{_fmt(p1.x)} {_fmt(p1.y)}")
            parts.append(
                f"C{_fmt(p2.x)} {_fmt(p2.y)} "
                f"{_fmt(p3.x)} {_fmt(p3.y)} {_fmt(p4.x)} {_fmt(p4.y)}"
            )
            current = p4
        elif kind == "re":
            rect = item[1]
            parts.append(
                f"M{_fmt(rect.x0)} {_fmt(rect.y0)}H{_fmt(rect.x1)}"
                f"V{_fmt(rect.y1)}H{_fmt(rect.x0)}Z"
            )
            current = None
        else:  # pragma: no cover - this rulebook contains only l/c/re paths
            raise ValueError(f"Unsupported path item type: {kind}")
    return " ".join(parts)


def _build_svg(doc, spec: dict) -> str:
    page = doc[spec["page"]]
    drawings = page.get_drawings()
    selected = [(drawings[index], color) for index, color in spec["drawings"]]

    rects = [drawing["rect"] for drawing, _ in selected]
    x0 = min(rect.x0 for rect in rects) - PAD_PT
    y0 = min(rect.y0 for rect in rects) - PAD_PT
    x1 = max(rect.x1 for rect in rects) + PAD_PT
    y1 = max(rect.y1 for rect in rects) + PAD_PT
    width, height = x1 - x0, y1 - y0

    elements = []
    for drawing, color in selected:
        rule = "evenodd" if drawing.get("even_odd") else "nonzero"
        elements.append(
            f'<path d="{_drawing_to_svg_path(drawing)}" '
            f'fill="{_hex_color(color)}" fill-rule="{rule}"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" '
        f'viewBox="{_fmt(x0)} {_fmt(y0)} {_fmt(width)} {_fmt(height)}">'
        + "".join(elements)
        + "</svg>"
    )


def _render_png(svg: str, output_path: Path) -> tuple[int, int]:
    """Rasterize the SVG with MuPDF at high resolution and trim empty edges."""
    svg_doc = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    page = svg_doc[0]
    zoom = TARGET_PX / max(page.rect.width, page.rect.height)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
    temp_path = output_path.with_suffix(".raw.png")
    pixmap.save(temp_path)

    with Image.open(temp_path) as image:
        rgba = image.convert("RGBA")
    temp_path.unlink(missing_ok=True)

    bbox = rgba.getchannel("A").getbbox()
    if bbox:
        rgba = rgba.crop(bbox)

    # Small transparent margin keeps nearest-neighbour scaling from clipping
    # antialiased edges against the texture behind the symbol.
    margin = 2
    padded = Image.new("RGBA", (rgba.width + margin * 2, rgba.height + margin * 2), (0, 0, 0, 0))
    padded.alpha_composite(rgba, (margin, margin))
    padded.save(output_path)
    return padded.size


def _make_contact_sheet(manifest: dict) -> None:
    sheet_path = SHEET_PATH
    sheet_path.parent.mkdir(parents=True, exist_ok=True)

    cell = 190
    label_h = 24
    cols = len(manifest["symbols"])
    sheet = Image.new("RGB", (cols * cell, cell * 2 + label_h), (225, 225, 225))
    draw = ImageDraw.Draw(sheet)

    for index, symbol in enumerate(manifest["symbols"]):
        asset_path = OUT_DIR / symbol["file"]
        with Image.open(asset_path) as asset:
            icon = asset.convert("RGBA")
            icon.thumbnail((cell - 34, cell - 34), Image.LANCZOS)
        for row, background in enumerate(((32, 32, 34, 255), (232, 232, 230, 255))):
            tile = Image.new("RGBA", (cell, cell), background)
            tile.alpha_composite(icon, ((cell - icon.width) // 2, (cell - icon.height) // 2))
            sheet.paste(tile.convert("RGB"), (index * cell, row * cell))
        x = index * cell
        draw.text((x + 6, cell * 2 + 6), f'{symbol["name"]}  ({symbol["file"]})', fill=(20, 20, 20))
        line_y = cell * 2 + 2
        draw.line([(x, line_y), (x + cell, line_y)], fill=(160, 160, 160))

    sheet.save(sheet_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF_PATH)

    rendered = {}
    for spec in ASSET_SPECS:
        svg = _build_svg(doc, spec)
        svg_path = OUT_DIR / f'{spec["file"]}.svg'
        svg_path.write_text(svg, encoding="utf-8")
        png_path = OUT_DIR / f'{spec["file"]}.png'
        size = _render_png(svg, png_path)
        rendered[spec["file"]] = png_path.stat().st_size
        print(f'{spec["file"]:<18} {size[0]}x{size[1]}px  {png_path.stat().st_size // 1024} KB')

    manifest = {
        "version": 1,
        "source": (
            "Primal: The Awakening rulebook; vector symbols from PDF pages 25, "
            "27 and 47 (printed pages 24, 25 and 41)"
        ),
        "resolution": TARGET_PX,
        "symbols": SYMBOLS,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")

    _make_contact_sheet(manifest)
    print(f"Wrote {SHEET_PATH}")


if __name__ == "__main__":
    main()
