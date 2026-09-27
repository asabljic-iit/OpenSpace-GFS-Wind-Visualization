from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
import glob
import os

app = FastAPI()

BASE_DIR = "./osfls_files"
BASE_URL = "http://localhost:8000/static"

os.makedirs(BASE_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

def get_target_dir(data_id: str) -> tuple[str, str]:
    """
    Cleans incoming data_id and returns (clean_data_id, target_directory_path).
    """
    # Strips any trailing query delimiters if OpenSpace appended them
    clean_id = str(data_id).split("?")[0].split("&")[0].strip()
    return clean_id, os.path.join(BASE_DIR, clean_id)

@app.get("/api/info/{data_id}")
def get_info(data_id: str):
    clean_id, target_dir = get_target_dir(data_id)
    files = sorted(glob.glob(os.path.join(target_dir, "*.osfls")))
    if not files:
        return {"datafeeds": []}
    
    first_file = os.path.basename(files[0])[:19]  # YYYY-MM-DDThh-mm-ss
    last_file = os.path.basename(files[-1])[:19]
    
    start_dt = datetime.strptime(first_file, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(last_file, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)

    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    return {
        "datafeeds": [
            {
                "id": clean_id,
                "startDate": start_str,
                "endDate": end_str,
                "cadence_in_seconds": 3600,
                "availability": {
                    "startDate": start_str,
                    "stopDate": end_str
                }
            }
        ]
    }

@app.get("/api/data")
@app.get("/api/data/")
def get_data(request: Request):
    query_str = request.url.query
    
    data_id = "1001"  # Default fallback ID if none specified
    time_min = None
    time_max = None
    
    # Parse parameter key-value pairs
    for param in query_str.split("&"):
        if param.startswith("time.min="):
            time_min = param.split("=")[1]
        elif param.startswith("time.max="):
            time_max = param.split("=")[1]
        elif param and "=" not in param:
            data_id = param  # Raw DataID passed by OpenSpace at start of query string

    if not time_min or not time_max:
        return {"files": []}

    clean_id, target_dir = get_target_dir(data_id)
    
    if not os.path.exists(target_dir):
        return {"files": []}

    clean_min = time_min.replace("Z", "").split(".")[0]
    clean_max = time_max.replace("Z", "").split(".")[0]
    
    req_start = datetime.strptime(clean_min, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    req_end = datetime.strptime(clean_max, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

    matched_files = []
    
    for filepath in glob.glob(os.path.join(target_dir, "*.osfls")):
        filename = os.path.basename(filepath)
        time_str = filename[:19]  # YYYY-MM-DDThh-mm-ss
        
        file_dt = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
        
        if req_start <= file_dt <= req_end:
            iso_timestamp = file_dt.strftime("%Y-%m-%d %H:%M:%S.0")
            file_url = f"{BASE_URL}/{clean_id}/{filename}"
            
            matched_files.append({
                "timestamp": iso_timestamp,
                "time": iso_timestamp,
                "url": file_url,
                "downloadUrl": file_url
            })

    return {"files": matched_files}