import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pymysql
except ImportError:
    pymysql = None

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from inference import pan_ocr_inference

app = FastAPI()

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import sqlite3
import threading
import time
import urllib.request

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pan_ocr.db")

# ── KEEP ALIVE WORKER ──
def _start_keep_alive():
    """Background daemon to ping public URL periodically and prevent Render idle spindown"""
    ping_url = os.environ.get("RENDER_EXTERNAL_URL", "https://pan-ocr-backend.onrender.com").rstrip("/") + "/health"
    def _worker():
        # Wait 3 minutes before beginning periodic pings
        time.sleep(180)
        while True:
            try:
                req = urllib.request.Request(
                    ping_url,
                    headers={"User-Agent": "Render-KeepAlive/1.0"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status == 200:
                        print(f"[KEEP-ALIVE] Heartbeat ping successful: {ping_url}")
            except Exception as e:
                print(f"[KEEP-ALIVE] Heartbeat error: {e}")
            # Ping every 10 minutes (600 seconds) - safely before Render's 15-minute idle limit
            time.sleep(600)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

if os.environ.get("ENABLE_KEEP_ALIVE", "true").lower() in ("true", "1", "yes"):
    _start_keep_alive()

# ── DB CONFIG ──
DB_CONFIG = {
    "host":     os.environ.get("MYSQL_HOST"),
    "user":     os.environ.get("MYSQL_USER",     "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "RootPass123"),
    "database": os.environ.get("MYSQL_DATABASE", "pan_ocr"),
    "port":     int(os.environ.get("MYSQL_PORT",  3306)),
    "cursorclass": pymysql.cursors.DictCursor if pymysql else None
}

def init_db():
    if os.environ.get("MYSQL_HOST") and pymysql:
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scan_results (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        pan_number VARCHAR(10),
                        name VARCHAR(255),
                        father_name VARCHAR(255),
                        dob VARCHAR(20),
                        is_pan_card BOOLEAN DEFAULT TRUE,
                        confidence FLOAT,
                        status VARCHAR(50),
                        error_msg TEXT,
                        scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()
            conn.close()
            return "mysql"
        except Exception as e:
            print(f"[MYSQL INIT ERROR - FALLING BACK TO SQLITE]: {e}")

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                pan_number TEXT,
                name TEXT,
                father_name TEXT,
                dob TEXT,
                is_pan_card BOOLEAN DEFAULT 1,
                confidence REAL,
                status TEXT,
                error_msg TEXT,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
        return "sqlite"
    except Exception as e:
        print(f"[SQLITE INIT ERROR]: {e}")
        return None

# Initialize on startup
init_db()

def save_to_db(results):
    db_mode = init_db()
    if not db_mode:
        return
    try:
        if db_mode == "mysql":
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                for r in results:
                    cursor.execute("""
                        INSERT INTO scan_results
                            (filename, pan_number, name, father_name,
                             dob, is_pan_card, confidence, status, error_msg)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        r.get("filename", ""),
                        r.get("pan_number", ""),
                        r.get("name", ""),
                        r.get("father_name", ""),
                        r.get("dob", ""),
                        r.get("is_pan_card", True),
                        r.get("confidence", 0.0),
                        r.get("status", "success"),
                        r.get("error", "")
                    ))
            conn.commit()
            conn.close()
        else:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            for r in results:
                cursor.execute("""
                    INSERT INTO scan_results
                        (filename, pan_number, name, father_name,
                         dob, is_pan_card, confidence, status, error_msg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get("filename", ""),
                    r.get("pan_number", ""),
                    r.get("name", ""),
                    r.get("father_name", ""),
                    r.get("dob", ""),
                    1 if r.get("is_pan_card", True) else 0,
                    r.get("confidence", 0.0),
                    r.get("status", "success"),
                    r.get("error", "")
                ))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[SAVE DB ERROR]: {e}")

# ── ROUTES ──

@app.get("/")
def root():
    return {
        "message": "PAN Card OCR Extraction System API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict",
            "predict_bulk": "/predict-bulk",
            "scans": "/scans"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        ocr_results = pan_ocr_inference(image_bytes)

        result = {
            "filename":    image.filename or "upload.jpg",
            "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
            "name":        ocr_results.get("Name",        {}).get("text", ""),
            "father_name": ocr_results.get("Father Name", {}).get("text", ""),
            "dob":         ocr_results.get("DOB",         {}).get("text", ""),
            "is_pan_card": True,
            "confidence":  0.95,
            "status":      "success"
        }

        save_to_db([result])
        return result

    except Exception as e:
        err = {
            "filename": image.filename or "upload.jpg",
            "status":   "error",
            "error":    str(e)
        }
        save_to_db([err])
        return err

@app.post("/predict-bulk")
async def predict_bulk(images: List[UploadFile] = File(...)):
    results = []

    for file in images:
        try:
            image_bytes = await file.read()
            ocr_results = pan_ocr_inference(image_bytes)

            results.append({
                "filename":    file.filename or "upload.jpg",
                "status":      "success",
                "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
                "name":        ocr_results.get("Name",        {}).get("text", ""),
                "father_name": ocr_results.get("Father Name", {}).get("text", ""),
                "dob":         ocr_results.get("DOB",         {}).get("text", ""),
                "is_pan_card": True,
                "confidence":  0.95
            })

        except Exception as e:
            results.append({
                "filename": file.filename or "upload.jpg",
                "status":   "error",
                "error":    str(e)
            })

    save_to_db(results)

    return {
        "total":   len(images),
        "success": len([r for r in results if r["status"] == "success"]),
        "failed":  len([r for r in results if r["status"] == "error"]),
        "results": results
    }

def fetch_scans(pan_number=None):
    db_mode = init_db()
    if db_mode == "mysql":
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            if pan_number:
                cur.execute("SELECT * FROM scan_results WHERE pan_number = %s ORDER BY scanned_at DESC", (pan_number,))
            else:
                cur.execute("SELECT * FROM scan_results ORDER BY scanned_at DESC")
            rows = cur.fetchall()
        conn.close()
        return rows
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if pan_number:
            cur.execute("SELECT * FROM scan_results WHERE pan_number = ? ORDER BY scanned_at DESC", (pan_number,))
        else:
            cur.execute("SELECT * FROM scan_results ORDER BY scanned_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

@app.get("/scans")
def get_scans():
    try:
        rows = fetch_scans()
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"total": 0, "scans": [], "error": str(e)}

@app.get("/scans/{pan_number}")
def get_scan_by_pan(pan_number: str):
    try:
        rows = fetch_scans(pan_number)
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"total": 0, "scans": [], "error": str(e)}

@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: int):
    try:
        db_mode = init_db()
        if db_mode == "mysql":
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM scan_results WHERE id = %s", (scan_id,))
            conn.commit()
            conn.close()
        else:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cur = conn.cursor()
            cur.execute("DELETE FROM scan_results WHERE id = ?", (scan_id,))
            conn.commit()
            conn.close()
        return {"message": f"Scan #{scan_id} deleted"}
    except Exception as e:
        return {"error": str(e)}

# ── RUN ──
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)