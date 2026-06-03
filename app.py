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
    """Resize image to a safe size for Tesseract on low-memory servers."""
    image = image.convert("RGB")
    w, h = image.size
    # Cap at 1000px wide — enough for OCR, safe for 512MB RAM
    if w > 1000:
        scale = 1000 / w
        image = image.resize((1000, int(h * scale)), Image.LANCZOS)
    elif w < 600:
        scale = 600 / w
        image = image.resize((600, int(h * scale)), Image.LANCZOS)
    return image


def extract_text(image: Image.Image) -> str:
    """Run Tesseract OCR and return cleaned text."""
    processed = preprocess_image(image)
    # PSM 11 = sparse text, finds text anywhere in the image
    raw = pytesseract.image_to_string(processed, config="--psm 11")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/test-ocr")
def test_ocr():
    """Generate a simple test image with known text and run OCR on it."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("L", (400, 100), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "FDU-X2", fill=0)
    result = pytesseract.image_to_string(img).strip()
    return jsonify({"test_text": "FDU-X2", "ocr_result": result, "match": "FDU" in result})


@app.route("/debug-ocr", methods=["POST"])
def debug_ocr():
    """Return the preprocessed image as base64 for inspection."""
    import base64
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    from PIL import ImageOps
    image = Image.open(io.BytesIO(request.files["image"].read()))
    image = ImageOps.exif_transpose(image)
    processed = preprocess_image(image)
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    raw = pytesseract.image_to_string(processed)
    return jsonify({
        "ocr_result": raw.strip(),
        "image_size": processed.size,
        "preview": f"data:image/png;base64,{b64[:200]}..."
    })


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
        from PIL import ImageOps
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.exif_transpose(image)  # fix iOS/Android rotation
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        text = extract_text(image)
    except Exception as e:
        import traceback
        return jsonify({"error": f"OCR failed: {str(e)}", "trace": traceback.format_exc()}), 500

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
        "image_size": f"{image.size[0]}x{image.size[1]}",
        "image_mode": image.mode,
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
