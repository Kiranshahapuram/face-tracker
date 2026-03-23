from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import psycopg2
import psycopg2.extras
import uvicorn
import os
import json
import uuid
import subprocess
from modules.config import Config

app = FastAPI(title="Face Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs('uploads', exist_ok=True)
os.makedirs('logs/entrs', exist_ok=True)
os.makedirs('logs/exits', exist_ok=True)

config = Config("config.json")

def get_db_connection():
    return psycopg2.connect(
        host=config.database.host,
        port=config.database.port,
        dbname=config.database.dbname,
        user=config.database.user,
        password=config.database.password if config.database.password else "postgres"
    )

@app.get("/api/dashboard/stats")
async def get_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT COUNT(*) as entry_count FROM events WHERE event_type = 'entry'")
        entry_count = cur.fetchone()['entry_count']
        
        cur.execute("SELECT COUNT(*) as exit_count FROM events WHERE event_type = 'exit'")
        exit_count = cur.fetchone()['exit_count']
        
        cur.execute("SELECT COUNT(*) as faces_count FROM faces")
        faces_count = cur.fetchone()['faces_count']
        
        cur.close()
        conn.close()
        return {
            "entry_count": entry_count,
            "exit_count": exit_count,
            "unique_visitors": faces_count
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sessions")
async def get_sessions():
    """Return per-video/session stats: unique visitors, entries, exits, duration."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT 
                s.id,
                s.video_source,
                s.unique_visitors,
                s.started_at,
                s.ended_at,
                (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.event_type = 'entry') as entry_count,
                (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.event_type = 'exit') as exit_count
            FROM sessions s
            ORDER BY s.started_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            started = r["started_at"]
            ended = r["ended_at"]
            duration_s = None
            if started and ended:
                duration_s = (ended - started).total_seconds()
            
            # Extract just the filename from the path
            video_name = os.path.basename(r["video_source"]) if r["video_source"] else "Unknown"
            
            result.append({
                "id": str(r["id"]),
                "video_source": r["video_source"],
                "video_name": video_name,
                "unique_visitors": r["unique_visitors"],
                "entry_count": r["entry_count"],
                "exit_count": r["exit_count"],
                "started_at": started,
                "ended_at": ended,
                "duration_s": duration_s
            })
        cur.close()
        conn.close()
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/faces")
async def get_faces():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get faces and their latest entry/exit images
        cur.execute("""
            SELECT f.id, 
                   f.first_seen, 
                   f.last_seen, 
                   f.visit_count,
                   (SELECT image_path FROM events WHERE face_id = f.id AND event_type = 'entry' ORDER BY occurred_at DESC LIMIT 1) as latest_entry_image,
                   (SELECT image_path FROM events WHERE face_id = f.id AND event_type = 'exit' ORDER BY occurred_at DESC LIMIT 1) as latest_exit_image
            FROM faces f
            ORDER BY f.last_seen DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        
        result = []
        for r in rows:
            result.append({
                "id": str(r["id"]),
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "visit_count": r["visit_count"],
                "latest_entry_image": r["latest_entry_image"] if r.get("latest_entry_image") else None,
                "latest_exit_image": r["latest_exit_image"] if r.get("latest_exit_image") else None
            })
            
        cur.close()
        conn.close()
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/events")
async def get_events(limit: int = 50):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute("""
            SELECT id, face_id, event_type, occurred_at, image_path, track_id, frame_number
            FROM events
            ORDER BY occurred_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "face_id": str(r["face_id"]),
                "event_type": r["event_type"],
                "occurred_at": r["occurred_at"],
                "image_path": r["image_path"],
                "track_id": r["track_id"],
                "frame_number": r["frame_number"]
            })
            
        cur.close()
        conn.close()
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/images/{path:path}")
async def get_image(path: str):
    # Ensure it only serves files from logs directory
    image_path = os.path.join(os.getcwd(), path)
    if os.path.exists(image_path) and os.path.isfile(image_path):
        return FileResponse(image_path)
    return {"error": "File not found"}

@app.post("/api/tracker/start")
async def start_tracker(
    source_type: str = Form(...), 
    rtsp_url: str = Form(""), 
    video_file: UploadFile = File(None),
    show_visualization: str = Form("true")
):
    filepath = ""
    if source_type == "rtsp" and rtsp_url:
        filepath = rtsp_url
    elif source_type == "upload" and video_file:
        file_location = f"uploads/{video_file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await video_file.read())
        filepath = file_location
    else:
        return {"error": "Invalid input"}

    # Update config.json
    with open("config.json", "r") as f:
        conf_data = json.load(f)
    
    conf_data["video"]["source"] = filepath
    
    if "system" not in conf_data:
        conf_data["system"] = {}
    conf_data["system"]["show_visualization"] = (show_visualization.lower() == "true")
    
    with open("config.json", "w") as f:
        json.dump(conf_data, f, indent=2)
        
    # Start the tracker using python subprocess or just run main.py in the background
    # Using start /B or similar on windows
    subprocess.Popen(["python", "main.py"], shell=True)
    
    return {"message": "Tracker started successfully", "source": filepath}

@app.post("/api/tracker/stop")
async def stop_tracker():
    """Stop the tracker by creating a stop flag file."""
    try:
        with open("stop_flag.txt", "w") as f:
            f.write("stop")
        return {"message": "Stop flag created"}
    except Exception as e:
        # Fallback: keep the crude way if file access fails
        os.system('taskkill /f /im python.exe /fi "WINDOWTITLE eq main.py"')
        return {"message": "Stop command sent (fallback)", "error": str(e)}

@app.get("/api/tracker/status")
async def get_tracker_status():
    """Check if main.py is currently running."""
    import psutil
    running = False
    try:
        for proc in psutil.process_iter(['cmdline']):
            cmd = proc.info.get('cmdline')
            if cmd and 'main.py' in ' '.join(cmd):
                running = True
                break
    except:
        pass
    return {"running": running}

@app.get("/api/logs")
async def list_logs():
    """List all available log files."""
    log_dir = config.system.log_dir
    files = []
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            fpath = os.path.join(log_dir, f)
            if os.path.isfile(fpath) and f.endswith(('.log', '.txt')):
                size = os.path.getsize(fpath)
                files.append({"name": f, "size_bytes": size})
    return files

@app.get("/api/logs/download/{filename}")
async def download_log(filename: str):
    """Download a specific log file."""
    # Sanitize: only allow filenames, no path traversal
    safe_name = os.path.basename(filename)
    log_path = os.path.join(os.getcwd(), config.system.log_dir, safe_name)
    if os.path.exists(log_path) and os.path.isfile(log_path):
        return FileResponse(log_path, filename=safe_name, media_type="text/plain")
    return {"error": "Log file not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
