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

# ── DB CONFIG ──
DB_CONFIG = {
    "host":     os.environ.get("MYSQL_HOST",     "mysql-service"),
    "user":     os.environ.get("MYSQL_USER",     "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "RootPass123"),
    "database": os.environ.get("MYSQL_DATABASE", "pan_ocr"),
    "port":     int(os.environ.get("MYSQL_PORT",  3306)),
    "cursorclass": pymysql.cursors.DictCursor if pymysql else None
}

def get_db():
    if pymysql is None:
        return None
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[DB CONNECT ERROR] {e}")
        return None

def save_to_db(results):
    try:
        conn = get_db()
        if not conn:
            return
        with conn.cursor() as cursor:
            for r in results:
                cursor.execute("""
                    INSERT INTO scan_results
                        (filename, pan_number, name, father_name,
                         dob, is_pan_card, confidence, status, error_msg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    r.get("filename",    ""),
                    r.get("pan_number",  ""),
                    r.get("name",        ""),
                    r.get("father_name", ""),
                    r.get("dob",         ""),
                    r.get("is_pan_card", True),
                    r.get("confidence",  0.0),
                    r.get("status",      "success"),
                    r.get("error",       "")
                ))
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        try:
            conn.close()
        except:
            pass

# ── ROUTES ──

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

@app.get("/scans")
def get_scans():
    try:
        conn = get_db()
        if not conn:
            return {"total": 0, "scans": []}
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM scan_results ORDER BY scanned_at DESC"
            )
            rows = cursor.fetchall()
        conn.close()
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"error": str(e)}

@app.get("/scans/{pan_number}")
def get_scan_by_pan(pan_number: str):
    try:
        conn = get_db()
        if not conn:
            return {"total": 0, "scans": []}
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM scan_results WHERE pan_number = %s ORDER BY scanned_at DESC",
                (pan_number,)
            )
            rows = cursor.fetchall()
        conn.close()
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: int):
    try:
        conn = get_db()
        if not conn:
            return {"message": "Database not connected"}
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM scan_results WHERE id = %s", (scan_id,)
            )
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