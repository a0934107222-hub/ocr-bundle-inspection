"""
OCR Bundle Inspection Server
- SQLite database with categories and part numbers
- Excel import (Category | Part Number columns)
- End-user can add/remove PNs per category
"""

import os
import io
import re
import sqlite3
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import google.generativeai as genai

app = Flask(__name__, static_folder="static")

DB_PATH = os.path.join(os.path.dirname(__file__), "parts.db")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS uploads (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                filename      TEXT NOT NULL,
                uploaded_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                imported_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS categories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT UNIQUE NOT NULL,
                upload_id INTEGER REFERENCES uploads(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS parts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
                label_text  TEXT NOT NULL,
                part_number TEXT NOT NULL,
                UNIQUE(category_id, label_text)
            );
        """)

init_db()

def get_or_create_category(conn, name, upload_id=None):
    name = name.strip()
    row = conn.execute("SELECT id FROM categories WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO categories(name, upload_id) VALUES(?,?)", (name, upload_id)
    )
    return cur.lastrowid

# ── OCR ───────────────────────────────────────────────────────────────────────

def gemini_read_label(image: Image.Image) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    w, h = image.size
    if w > 1000:
        image = image.resize((1000, int(h * 1000 / w)), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    prompt = (
        "This is a photo of a jig board label used in electronics manufacturing. "
        "Read ALL text visible on the white label sticker. "
        "Return only the exact text, nothing else. "
        "If multiple lines, separate with a space."
    )
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}
    ])
    return response.text.strip()

# ── Routes: static ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "gemini": bool(GEMINI_API_KEY)})

# ── Routes: uploads ──────────────────────────────────────────────────────────

@app.route("/uploads", methods=["GET"])
def list_uploads():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.id, u.filename, u.uploaded_at, u.imported_count,
                   GROUP_CONCAT(c.name, '||') as categories
            FROM uploads u
            LEFT JOIN categories c ON c.upload_id = u.id
            GROUP BY u.id
            ORDER BY u.uploaded_at DESC
        """).fetchall()
    result = []
    for r in rows:
        cats = [c for c in (r["categories"] or "").split("||") if c]
        result.append({
            "id":             r["id"],
            "filename":       r["filename"],
            "uploaded_at":    r["uploaded_at"],
            "imported_count": r["imported_count"],
            "categories":     cats
        })
    return jsonify(result)

@app.route("/uploads/<int:upload_id>", methods=["DELETE"])
def delete_upload(upload_id):
    with get_db() as conn:
        # Delete categories linked to this upload (cascades to parts)
        conn.execute("DELETE FROM categories WHERE upload_id=?", (upload_id,))
        conn.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    return jsonify({"message": "deleted"})

# ── Routes: categories ────────────────────────────────────────────────────────

@app.route("/categories", methods=["GET"])
def list_categories():
    with get_db() as conn:
        rows = conn.execute("SELECT id, name FROM categories ORDER BY name").fetchall()
    return jsonify([{"id": r["id"], "name": r["name"]} for r in rows])

@app.route("/categories", methods=["POST"])
def add_category():
    name = request.get_json(force=True).get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    with get_db() as conn:
        get_or_create_category(conn, name)
    return jsonify({"message": "ok", "name": name})

@app.route("/categories/<int:cat_id>", methods=["DELETE"])
def delete_category(cat_id):
    with get_db() as conn:
        conn.execute("DELETE FROM categories WHERE id=?", (cat_id,))
    return jsonify({"message": "deleted"})

# ── Routes: parts ─────────────────────────────────────────────────────────────

@app.route("/parts", methods=["GET"])
def list_parts():
    cat_id   = request.args.get("category_id")
    upload_id = request.args.get("upload_id")
    with get_db() as conn:
        if upload_id:
            rows = conn.execute("""
                SELECT p.label_text, p.part_number
                FROM parts p
                JOIN categories c ON p.category_id = c.id
                WHERE c.upload_id = ?
                ORDER BY p.label_text
            """, (upload_id,)).fetchall()
        elif cat_id:
            rows = conn.execute(
                "SELECT label_text, part_number FROM parts WHERE category_id=? ORDER BY label_text",
                (cat_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT label_text, part_number FROM parts ORDER BY label_text"
            ).fetchall()
    return jsonify([{"label_text": r["label_text"], "part_number": r["part_number"]} for r in rows])

@app.route("/parts", methods=["POST"])
def add_part():
    data     = request.get_json(force=True)
    label    = data.get("label_text", "").strip().upper()
    pn       = data.get("part_number", "").strip().upper()
    cat_name = data.get("category", "General").strip()
    # Allow label_text == part_number if only part_number is given
    if not label:
        label = pn
    if not pn:
        return jsonify({"error": "part_number required"}), 400
    with get_db() as conn:
        cat_id = get_or_create_category(conn, cat_name)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO parts(category_id, label_text, part_number) VALUES(?,?,?)",
                (cat_id, label, pn)
            )
        except sqlite3.IntegrityError:
            pass
    return jsonify({"message": "ok", "label_text": label, "part_number": pn})

@app.route("/parts/<path:label>", methods=["DELETE"])
def delete_part(label):
    cat_id = request.args.get("category_id")
    with get_db() as conn:
        if cat_id:
            conn.execute("DELETE FROM parts WHERE label_text=? AND category_id=?", (label.upper(), cat_id))
        else:
            conn.execute("DELETE FROM parts WHERE label_text=?", (label.upper(),))
    return jsonify({"message": "deleted"})

# ── Routes: Excel import ──────────────────────────────────────────────────────

@app.route("/upload-excel", methods=["POST"])
def upload_excel():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(request.files["file"].read()), data_only=True)
        ws = wb.active
    except Exception as e:
        return jsonify({"error": f"Cannot read Excel: {e}"}), 400

    filename = request.files["file"].filename or "unknown.xlsx"
    imported = 0
    errors   = []

    with get_db() as conn:
        # Create upload record first
        cur = conn.execute(
            "INSERT INTO uploads(filename, imported_count) VALUES(?,0)", (filename,)
        )
        upload_id = cur.lastrowid

        for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            if not row or row[0] is None:
                continue
            label    = str(row[0]).strip().upper() if row[0] else ""
            pn       = str(row[1]).strip().upper() if len(row) > 1 and row[1] else ""
            category = str(row[2]).strip()         if len(row) > 2 and row[2] else "General"
            if not label:
                errors.append(f"Row {i}: missing label text")
                continue
            if not pn:
                pn = label
            cat_id = get_or_create_category(conn, category, upload_id)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO parts(category_id, label_text, part_number) VALUES(?,?,?)",
                    (cat_id, label, pn)
                )
                imported += 1
            except Exception as ex:
                errors.append(f"Row {i}: {ex}")

        # Update the count
        conn.execute("UPDATE uploads SET imported_count=? WHERE id=?", (imported, upload_id))

    return jsonify({"imported": imported, "upload_id": upload_id, "errors": errors})

# ── Routes: OCR ───────────────────────────────────────────────────────────────

@app.route("/ocr", methods=["POST"])
def ocr():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500
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
        return jsonify({"error": f"Gemini OCR failed: {e}"}), 500

    tokens = re.findall(r"[A-Z0-9][A-Z0-9_\-\.]{2,}", text.upper())
    best   = max(tokens, key=len) if tokens else text.strip().upper()

    upload_id = request.form.get("upload_id")
    with get_db() as conn:
        if upload_id:
            row = conn.execute("""
                SELECT p.label_text, p.part_number FROM parts p
                JOIN categories c ON p.category_id = c.id
                WHERE c.upload_id = ? AND p.label_text = ?
            """, (upload_id, best)).fetchone()
        else:
            row = conn.execute(
                "SELECT label_text, part_number FROM parts WHERE label_text=?", (best,)
            ).fetchone()
    passed = row is not None
    matched_pn = row["part_number"] if row else None
    return jsonify({"raw_text": text, "detected": best, "passed": passed, "part_number": matched_pn})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"  OCR Bundle Inspection — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
