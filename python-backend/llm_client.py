# llm_client.py
import base64
import io
import json
import re
import requests
from PIL import Image
from rag_client import query_rules

LLAMA_SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"
MODELS_URL = "http://127.0.0.1:9931/v1/models"

# Vision requests need a larger input than the tiny source card.  The cap
# keeps a full-resolution scan from flooding the local model.
_VISION_MAX_SIDE = 1400


def get_loaded_model():
    try:
        response = requests.get(MODELS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return data["data"][0].get("id", "unknown")
        return None
    except Exception as e:
        print(f"Could not fetch model info: {e}")
        return None


def get_vision_model():
    """Return the first loaded vision-language model, if the server has one."""
    try:
        response = requests.get(MODELS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Could not fetch vision model info: {e}")
        return None

    models = data.get("data") or []
    for model in models:
        model_id = str(model.get("id", ""))
        lowered = model_id.lower()
        if "vl" in lowered or "vision" in lowered or "llava" in lowered:
            return model_id
    return None


def _parse_json_response(text):
    """Extract the first JSON object/array, tolerating markdown fences."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    decoder = json.JSONDecoder()
    for opener in ("{", "["):
        start = cleaned.find(opener)
        if start < 0:
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def vision_json(image_path, system_prompt, user_prompt, model=None, max_tokens=512, timeout=120):
    """Send one image plus prompts to a local vision model and parse JSON.

    Returns ``(data, meta)``.  ``data`` is the parsed JSON value or None;
    ``meta`` carries the sent image dimensions and an error string.  This
    function never raises, so callers can fall back to template matching.
    """
    meta = {"width": 0, "height": 0, "error": ""}
    model_name = model or get_vision_model()
    if not model_name:
        meta["error"] = "no vision model loaded"
        return None, meta

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest < 720:
                factor = 2
            elif longest > _VISION_MAX_SIDE:
                factor = _VISION_MAX_SIDE / longest
            else:
                factor = 1.0
            if factor != 1:
                image = image.resize(
                    (max(1, round(width * factor)), max(1, round(height * factor))),
                    Image.LANCZOS,
                )
            meta["width"], meta["height"] = image.size
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
    except Exception as e:
        meta["error"] = f"could not read image: {e}"
        return None, meta

    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        meta["error"] = f"vision request failed: {e}"
        return None, meta

    parsed = _parse_json_response(content)
    if parsed is None:
        preview = (content or "").strip().replace("\n", " ")[:200]
        meta["error"] = f"model did not return valid JSON: {preview}"
        return None, meta
    return parsed, meta


def fix_ocr_text(raw_text_fragments, card_name="Unknown Card", return_context=False):
    """Fixes OCR text via the local llama.cpp server.

    Returns corrected text, or (corrected, context) when return_context is True.
    Never raises; returns (None, context) / None on failure.
    """
    context = ""
    corrected_text = None
    model_name = get_loaded_model()

    if model_name:
        combined_text = "\n".join(raw_text_fragments)

        # 1. Query the RAG database for rulebook context
        search_query = f"Card layout and rules for text: {combined_text}. What do symbols mean?"
        context, _chunks = query_rules(search_query)

        # 2. Build the system prompt with the RAG context wrapped in XML delimiters
        system_prompt = (
            "You are an expert OCR correction assistant for the card game 'Primal: The Awakening'.\n"
            "<rulebook_context>\n"
            f"{context}\n"
            "</rulebook_context>\n"
            "The user will provide fragmented, typo-ridden text extracted from a card image, separated by newlines.\n"
            "Your job is to:\n"
            "1. Fix spelling errors (e.g., 'COUFDINATED' -> 'COORDINATED', 'Chcos' -> 'Choose').\n"
            "2. Replace numbers or typos that are clearly meant to be inline symbols with the EXACT string '[SYMBOL]'.\n"
            "3. Use the provided Rulebook Context to ensure you use the exact correct terminology for this game.\n"
            "4. CRITICAL: Preserve the EXACT line breaks from the input. Do NOT merge lines into a single paragraph.\n"
            "Output ONLY the corrected text. Do not add markdown formatting, backticks, or conversational text.\n"
            "CRITICAL SYMBOL RULE: Replace inline symbol placeholders with the EXACT string '[SYMBOL]'. "
            "Never use '[Symbol]', '[symbol]', or any other casing. Only '[SYMBOL]', all caps, inside square brackets."
        )

        # The raw OCR text is delivered inside an XML-delimited block, never concatenated raw.
        user_content = f"<ocr_text>\n{combined_text}\n</ocr_text>"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
            "stream": False
        }

        try:
            response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            corrected_text = result['choices'][0]['message']['content'].strip()
            corrected_text = corrected_text.replace("```text", "").replace("```", "").strip()
            corrected_text = re.sub(r"\[symbol\]", "[SYMBOL]", corrected_text, flags=re.IGNORECASE).strip()

        except Exception as e:
            print(f"LLM error: {e}")

    if return_context:
        return (corrected_text, context)
    return corrected_text
