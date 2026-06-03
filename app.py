"""
OCR Bundle Inspection Server
Uses Google Gemini Vision AI to read label text and validate against approved parts list.
"""

import os
import csv
import re
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import google.generativeai as genai

app = Flask(__name__, static_folder="static")

PARTS_LIST_FILE = os.path.join(os.path.dirname(__file__), "parts_list.csv")

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_parts_list():
    parts = set()
    if os.path.exists(PARTS_LIST_FILE):
        with open(PARTS_LIST_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pn = row.get("part_number", "").strip().upper()
                if pn:
                    parts.add(pn)
    return parts


def gemini_read_label(image: Image.Image) -> str:
    """Use Gemini Vision to read text from a label image."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Resize to max 1000px to reduce token usage
    w, h = image.size
    if w > 1000:
        image = image.resize((1000, int(h * 1000 / w)), Image.LANCZOS)

    # Convert to JPEG bytes
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    prompt = (
        "This is a photo of a jig board label used in electronics manufacturing. "
        "Read ALL text visible on the white label sticker in the image. "
        "Return only the exact text you see on the label, nothing else. "
        "If there are multiple lines, separate them with a space."
    )

    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}
    ])
    return response.text.strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    if not GEMINI_API_KEY:
        return jsonify({"status": "error", "detail": "GEMINI_API_KEY not set"}), 500
    return jsonify({"status": "ok", "model": "gemini-2.0-flash"})


@app.route("/ocr", methods=["POST"])
def ocr():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not configured on server"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read()))
        image = ImageOps.exif_transpose(image)
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        text = gemini_read_label(image)
    except Exception as e:
        return jsonify({"error": f"Gemini OCR failed: {str(e)}"}), 500

    # Extract part number tokens
    tokens = re.findall(r"[A-Z0-9][A-Z0-9_\-\.]{2,}", text.upper())
    best_match = max(tokens, key=len) if tokens else text.strip().upper()

    parts = load_parts_list()
    passed = best_match in parts

    return jsonify({
        "raw_text": text,
        "detected": best_match,
        "passed": passed,
    })


@app.route("/parts", methods=["GET"])
def list_parts():
    return jsonify(sorted(load_parts_list()))


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
        csv.writer(f).writerow([pn])
    return jsonify({"message": "Added", "part_number": pn})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  OCR Bundle Inspection Server (Gemini Vision)")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
