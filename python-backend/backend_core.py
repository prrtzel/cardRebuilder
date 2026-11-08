import cv2
import json
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONTS_DIR = Path(__file__).parent / "fonts"


class CardProcessor:
    def __init__(self, inpaint_radius: int = 3):
        # Store config only; the OCR reader is lazy-loaded on first use so
        # that neither the server nor the UI ever pay the EasyOCR model-load
        # cost at startup.
        self.inpaint_radius = inpaint_radius
        self.reader = None

    def _ensure_reader(self):
        """Lazily initialize the EasyOCR reader on first OCR call."""
        if self.reader is None:
            print("Loading OCR Model...")
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("OCR Model Loaded.")
        return self.reader

    def preprocess_for_ocr(self, image_path):
        """Upscales and thresholds the image for better OCR reading."""
        img = cv2.imread(image_path)

        # 1. Upscale by 3x (Lanczos is great for keeping edges sharp)
        img_large = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)

        # 3. Apply Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 15)

        return img_large, thresh

    def extract_text(self, image_path, output_json="ocr_data.json"):
        """Finds all text and bounding boxes from scratch."""
        reader = self._ensure_reader()
        print(f"Extracting text from {image_path}...")

        img_large, thresh = self.preprocess_for_ocr(image_path)
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        results = reader.readtext(thresh_rgb, detail=1)

        extracted_data = []
        for (bbox, text, prob) in results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            # Divide coordinates by 3 to map them back to the original image size
            item = {
                "text": text,
                "confidence": float(prob),
                "bbox": {
                    "x_min": int(min(x_coords) / 3),
                    "y_min": int(min(y_coords) / 3),
                    "x_max": int(max(x_coords) / 3),
                    "y_max": int(max(y_coords) / 3)
                }
            }
            extracted_data.append(item)

        with open(output_json, 'w') as f:
            json.dump(extracted_data, f, indent=4)

        print(f"Text extracted and saved to {output_json}")
        return extracted_data

    def _fit_font(self, draw, text, font_path, max_width, max_height):
        """Binary-search the largest font size fitting text into max_width/max_height."""
        lo = 6
        hi = max(8, int(max_height * 1.2))
        best_size = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(str(font_path), mid)
            bbox = draw.textbbox((0, 0), text, font=font)
            if (bbox[2] - bbox[0]) <= max_width and (bbox[3] - bbox[1]) <= max_height:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ImageFont.truetype(str(font_path), best_size)

    def _wrap_text(self, draw, text, font, max_width):
        """Wrap text on whitespace boundaries; never emit empty lines."""
        words = text.split()
        if not words:
            return [text] if text.strip() else [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    @staticmethod
    def _fit_symbol_placement(asset_size, bbox, scale=1.0):
        """Fits golden artwork inside a bbox and returns pixel placement.

        The asset keeps its aspect ratio and is centered in the box.  The
        result is expressed in ``scale``d image coordinates so the inpaint
        mask (scale 1) and golden compositing (``coord_scale``) share the
        same fit math and stay aligned.
        """
        asset_w, asset_h = asset_size
        x1 = float(bbox["x_min"]) * scale
        y1 = float(bbox["y_min"]) * scale
        x2 = float(bbox["x_max"]) * scale
        y2 = float(bbox["y_max"]) * scale
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        ratio = min(box_w / asset_w, box_h / asset_h)
        target_w = max(1, int(round(asset_w * ratio)))
        target_h = max(1, int(round(asset_h * ratio)))
        pos_x = int(round(x1 + (box_w - target_w) / 2))
        pos_y = int(round(y1 + (box_h - target_h) / 2))
        return pos_x, pos_y, target_w, target_h

    def inpaint_card(self, image_path, ocr_data, output_path="clean_background.png", extra_boxes=None, symbols=None):
        """Removes text (and replaceable golden symbols) from the image.

        Items with ``inpaint`` set to False are skipped, ``extra_boxes``
        (manually drawn rectangles) are added to the mask as-is, and symbols
        are masked using the golden asset's silhouette when available so the
        banner, frame, and artwork around the glyph survive.
        """
        print("Inpainting card (removing text and symbols)...")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = img.shape[:2]
        text_mask = np.zeros((height, width), dtype=np.uint8)
        symbol_mask = np.zeros((height, width), dtype=np.uint8)
        masked = 0
        masked_symbols = 0

        for item in ocr_data:
            if item.get("inpaint", True) is False:
                continue
            bbox = item['bbox']
            # Dynamic padding: scale with the box height, clamped to [2, 8]
            h = bbox['y_max'] - bbox['y_min']
            padding = int(max(2, min(h * 0.15, 8)))
            x1 = max(0, bbox['x_min'] - padding)
            y1 = max(0, bbox['y_min'] - padding)
            x2 = min(img.shape[1], bbox['x_max'] + padding)
            y2 = min(img.shape[0], bbox['y_max'] + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, -1)
            masked += 1

        for box in extra_boxes or []:
            x1 = max(0, min(img.shape[1], int(box['x_min'])))
            y1 = max(0, min(img.shape[0], int(box['y_min'])))
            x2 = max(0, min(img.shape[1], int(box['x_max'])))
            y2 = max(0, min(img.shape[0], int(box['y_max'])))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, -1)
            masked += 1

        library = None
        for item in symbols or []:
            if item.get("redraw", True) is False or not item.get("symbol_id"):
                continue
            bbox = item.get("bbox") or {}
            try:
                x1 = int(bbox['x_min'])
                y1 = int(bbox['y_min'])
                x2 = int(bbox['x_max'])
                y2 = int(bbox['y_max'])
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            if library is None:
                from symbols import get_symbol_library

                library = get_symbol_library()
            asset = library.pil_asset(item["symbol_id"])

            if asset is not None:
                pos_x, pos_y, target_w, target_h = self._fit_symbol_placement(
                    asset.size, bbox
                )
                rx1 = max(0, pos_x)
                ry1 = max(0, pos_y)
                rx2 = min(width, pos_x + target_w)
                ry2 = min(height, pos_y + target_h)
                if target_w >= 3 and target_h >= 3 and rx2 > rx1 and ry2 > ry1:
                    # Mask only the glyph's silhouette.  The 1px dilation
                    # covers the printed symbol's antialiased fringe while
                    # preserving banner/frame pixels just outside the glyph.
                    alpha = asset.getchannel("A").resize(
                        (target_w, target_h), Image.LANCZOS
                    )
                    shape_mask = (np.asarray(alpha, dtype=np.uint8) >= 32)
                    shape_mask = (shape_mask.astype(np.uint8)) * 255
                    shape_mask = cv2.dilate(
                        shape_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1,
                    )
                    sub = shape_mask[ry1 - pos_y:ry2 - pos_y, rx1 - pos_x:rx2 - pos_x]
                    region = symbol_mask[ry1:ry2, rx1:rx2]
                    np.maximum(region, sub, out=region)
                    masked_symbols += 1
                    continue

            # Fallback for a missing or unusable asset: mask the tight bbox.
            fx1 = max(0, x1)
            fy1 = max(0, y1)
            fx2 = min(width, x2)
            fy2 = min(height, y2)
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            cv2.rectangle(symbol_mask, (fx1, fy1), (fx2, fy2), 255, -1)
            masked_symbols += 1

        print(
            f"Inpainting {masked} text/region mask(s) and "
            f"{masked_symbols} symbol silhouette(s)..."
        )
        clean_img = cv2.inpaint(
            img, text_mask, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA
        )
        if masked_symbols:
            # A thinner radius is enough now that the mask hugs the glyph.
            clean_img = cv2.inpaint(
                clean_img, symbol_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA
            )
        cv2.imwrite(output_path, clean_img)
        print(f"Clean background saved to {output_path}")
        return output_path

    def render_text(self, background_path, ocr_data, output_path="rebuilt_card.png", coord_scale=1):
        """Draws new text over the clean background using PIL.

        ``coord_scale`` maps OCR boxes (original-image coordinates) onto a
        background that has been upscaled by that factor. Font selection still
        uses the original box heights so styling is identical at any scale.
        """
        bg = Image.open(background_path).convert("RGBA")
        draw = ImageDraw.Draw(bg)

        title_font_path = FONTS_DIR / "BebasNeue-Regular.ttf"
        body_font_path = FONTS_DIR / "Roboto-Regular.ttf"

        scale = float(coord_scale)
        pad = max(1, int(round(2 * scale)))
        min_width = max(1, int(round(10 * scale)))
        min_height = max(1, int(round(8 * scale)))

        for item in ocr_data:
            if item.get("redraw", True) is False:
                continue
            text = item.get("text", "")
            if not text:
                continue

            bbox = item['bbox']
            box_h = bbox['y_max'] - bbox['y_min']

            # Title vs body decided by box height, not absolute Y position.
            # The threshold is in original pixels so upscaling does not change
            # which font a line receives.
            if box_h > 30:
                font_path = title_font_path
                color = (40, 60, 150)  # Blue
            else:
                font_path = body_font_path
                color = (0, 0, 0)  # Black

            x_min = bbox['x_min'] * scale
            y_min = bbox['y_min'] * scale
            box_w = (bbox['x_max'] - bbox['x_min']) * scale

            max_width = max(int(round(box_w)), min_width)
            max_height = max(int(round(box_h * scale)), min_height)

            font = self._fit_font(draw, text, font_path, max_width, max_height)
            lines = self._wrap_text(draw, text, font, max_width)

            y = y_min - pad
            for line in lines:
                if not line:
                    continue
                draw.text((x_min, y), line, font=font, fill=color)
                top = draw.textbbox((0, 0), line, font=font)[1]
                bottom = draw.textbbox((0, 0), line, font=font)[3]
                y = y + (bottom - top) + pad

        bg.save(output_path)
        print(f"Rebuilt card saved to {output_path}")
        return output_path

    def render_symbols(self, background_path, symbols, output_path=None, coord_scale=1):
        """Draws golden symbol artwork over the rebuilt background.

        ``coord_scale`` maps the original card coordinates onto the upscaled
        background, exactly like :meth:`render_text`.  Only rows with a
        confirmed ``symbol_id`` and ``redraw`` enabled are composited.
        """
        from symbols import get_symbol_library

        if output_path is None:
            output_path = background_path
        library = get_symbol_library()
        with Image.open(background_path) as source:
            bg = source.convert("RGBA")
        scale = float(coord_scale)
        drawn = 0

        for item in symbols or []:
            if item.get("redraw", True) is False:
                continue
            symbol_id = item.get("symbol_id")
            if not symbol_id:
                continue
            asset = library.pil_asset(symbol_id)
            if asset is None:
                print(f"Unknown golden symbol id: {symbol_id}")
                continue
            bbox = item.get("bbox") or {}
            try:
                pos_x, pos_y, target_w, target_h = self._fit_symbol_placement(
                    asset.size, bbox, scale
                )
            except (KeyError, TypeError, ValueError):
                continue
            resized = asset.resize((target_w, target_h), Image.LANCZOS)
            bg.alpha_composite(resized, (pos_x, pos_y))
            drawn += 1

        bg.save(output_path)
        print(f"Drew {drawn} golden symbol(s) into {output_path}")
        return drawn
