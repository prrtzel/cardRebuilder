# Project Context: Primal Card Rebuilder

## Overview
This project is a tool for reconstructing and "rebuilding" cards for the card game *Primal: The Awakening*. It automates the process of identifying elements on a card, cleaning the card's background, and re-rendering text and symbols in a consistent style.

## Tech Stack
- **Language**: Python (Backend), Rust/Tauri (Frontend)
- **AI/ML**: 
    - `easyocr` for primary text extraction.
    - `llama.cpp` (via local server) for OCR correction and text generation.
    - Vision-Language Models (e.g., Qwen-VL, LLaVA) for symbol identification.
    - `chromadb` and `sentence-transformers` for RAG (Retrieval-Augmented Generation) using the game's rulebook.
- **Image Processing**: `OpenCV` (cv2), `Pillow` (PIL), `NumPy`.

## Core Logic & Workflow
1. **Ingestion & OCR**:
    - Images are upscaled (3x Lanczos) and thresholded for OCR.
    - `easyocr` extracts raw text fragments and bounding boxes.
    - `llm_client.py` sends these fragments to a local LLM.
    - **RAG Integration**: The LLM is provided with rulebook context via `rag_client.py` to ensure terminology is correct and typos are fixed.
    - **Symbol Rule**: The LLM is instructed to replace inline symbols with the `[SYMBOL]` placeholder.

2. **Symbol Detection (`symbols.py`)**:
    - **Hybrid Approach**: 
        - **AI First**: A Vision model identifies symbol IDs from a catalog.
        - **Template Refinement**: Template matching (using `cv2.matchTemplate`) is used to refine the coordinates and ensure the "golden" symbols are placed precisely.
        - **Fallback**: If the AI fails, a strict template-only scan is performed.
    - Symbols are categorized (offensive, defensive, etc.) and have specific size ranges for detection.

3. **Inpainting (`backend_core.py`)**:
    - The tool creates masks for text and symbols.
    - **Symbol Silhouette Masking**: Instead of masking a simple box for symbols, it uses the alpha channel of the "golden" asset to mask only the glyph itself, preserving the surrounding frame/banner.
    - `cv2.inpaint` (TELEA algorithm) is used to fill in the masked areas.

4. **Rendering**:
    - **Text**: Drawn using PIL. It automatically selects fonts (Title vs. Body) based on the height of the original OCR box. It includes a binary search to fit the largest possible font into the original bounding box.
    - **Symbols**: "Golden" assets are resized and alpha-composited onto the clean background using the refined coordinates.

## Key Files
- `python-backend/backend_core.py`: Main processing pipeline.
- `python-backend/llm_client.py`: AI communication and OCR correction logic.
- `python-backend/symbols.py`: Symbol library, template matching, and AI detection logic.
- `python-backend/rag_client.py`: ChromaDB integration for rulebook context.
- `symbols/`: Directory of "golden" symbol PNGs and the `manifest.json`.

## Development Notes
- **Coordinate Scaling**: The system works with "original image" coordinates. When rendering on an upscaled background, a `coord_scale` factor is used to map boxes correctly.
- **Lazy Loading**: The OCR model is lazy-loaded to prevent long startup times for the server/UI.
