"""
OCR Bundle Inspection Server
Reads label text from images and validates against an approved parts list.
"""

import os
import csv
import re
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import io

app = Flask(__name__, static_folder="static")

PARTS_LIST_FILE = os.path.join(os.path.dirname(__file__), "parts_list.csv")

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_parts_list():
    """Load approved part numbers from CSV."""
    parts = set()
    if os.path.exists(PARTS_LIST_FILE):
        with open(PARTS_LIST_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pn = row.get("part_number", "").strip().upper()
                if pn:
                    parts.add(pn)
    return parts


def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image for better OCR accuracy on printed labels."""
    image = image.convert("L")  # grayscale

    # Scale up so label text is at least 800px wide
    w, h = image.size
    if w < 1600:
        scale = 1600 / w
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Adaptive binarization: much cleaner than contrast boost for printed text
    import numpy as np
    arr = np.array(image)
    # Compute local mean with a large kernel (simulates adaptive threshold)
    from PIL import ImageFilter as IF
    blurred = image.filter(IF.GaussianBlur(radius=15))
    blur_arr = np.array(blurred).astype(int)
    # Pixels darker than local mean - 10 become black (text), rest white
    binary = ((arr.astype(int) - blur_arr) < -10).astype(np.uint8) * 255
    image = Image.fromarray(binary.astype(np.uint8))

    return image


def extract_text(image: Image.Image) -> str:
    """Run Tesseract OCR with multiple PSM modes, return best result."""
    processed = preprocess_image(image)

    best = ""
    for psm in [6, 11, 3]:
        config = f"--psm {psm} --oem 3"
        raw = pytesseract.image_to_string(processed, config=config)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
        text = "\n".join(ln for ln in lines if ln)
        if len(text) > len(best):
            best = text

    return best


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    """Check if Tesseract is installed and working."""
    try:
        version = pytesseract.get_tesseract_version()
        return jsonify({"status": "ok", "tesseract_version": str(version)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        text = extract_text(image)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    # Find the best matching token (longest token that looks like a part number)
    tokens = re.findall(r"[A-Z0-9][A-Z0-9_\-\.]{3,}", text.upper())
    best_match = max(tokens, key=len) if tokens else text.strip().upper()

    parts = load_parts_list()
    passed = best_match in parts

    return jsonify({
        "raw_text": text,
        "detected": best_match,
        "passed": passed,
        "total_parts": len(parts),
    })


@app.route("/parts", methods=["GET"])
def list_parts():
    parts = load_parts_list()
    return jsonify(sorted(parts))


@app.route("/parts", methods=["POST"])
def add_part():
    data = request.get_json(force=True)
    pn = data.get("part_number", "").strip().upper()
    if not pn:
        return jsonify({"error": "part_number required"}), 400

    parts = load_parts_list()
    if pn in parts:
        return jsonify({"message": "Already exists", "part_number": pn})

    with open(PARTS_LIST_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([pn])

    return jsonify({"message": "Added", "part_number": pn})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  OCR Bundle Inspection Server")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
