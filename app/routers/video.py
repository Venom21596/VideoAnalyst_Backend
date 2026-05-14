from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.core_engine import process_video
from app.topic_engine import generate_topic_timeline
from app.state import video_memory, embed_model
import shutil, os, re, time, subprocess

router = APIRouter()

# Accept ALL video formats
ALLOWED_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flv', '.wmv', '.m4v', '.3gp', '.ts',
    '.mpeg', '.mpg', '.ogv', '.rm', '.rmvb',
    '.divx', '.f4v', '.asf', '.vob'
}

def convert_to_mp4(input_path: str) -> str:
    """Convert any video format to mp4 using FFmpeg."""
    output_path = input_path.rsplit('.', 1)[0] + '_converted.mp4'
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return output_path

def is_mp4_compatible(file_path: str) -> bool:
    """Check if file needs conversion."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in {'.mp4', '.webm'}

@router.post("/process-video")
async def process(file: UploadFile = File(...)):
    filename = file.filename or "unnamed_video"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    os.makedirs("data", exist_ok=True)
    safe_name = re.sub(r'[^\w\-.]', '_', os.path.basename(filename))
    file_path = f"data/{safe_name}"

    # Handle duplicate filenames
    base, extension = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"data/{base}_{counter}{extension}"
        counter += 1

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Convert to mp4 if needed
    processing_path = file_path
    converted = False
    if not is_mp4_compatible(file_path):
        try:
            processing_path = convert_to_mp4(file_path)
            converted = True
        except RuntimeError as e:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail=str(e))

    try:
        data = process_video(processing_path)
        data['timestamp'] = time.time()
        data['file_path'] = processing_path
        data['original_format'] = ext
        data['topics'] = generate_topic_timeline(data['segments'], embed_model)

        video_id = os.path.basename(file_path)
        video_memory[video_id] = data

        return {
            "status": "success",
            "video_id": video_id,
            "original_format": ext,
            "converted_to_mp4": converted,
            "has_audio": data.get("has_audio", True),
            "analysis_method": "whisper" if data.get("has_audio", True) else "llava_visual",
            "chapters": data["chapters"],
            "topics": data["topics"],
            "segment_count": len(data["segments"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos")
async def list_videos():
    return {
        "videos": [
            {
                "video_id": vid,
                "timestamp": data.get("timestamp"),
                "original_format": data.get("original_format", "unknown"),
                "topics": data.get("topics", [])
            }
            for vid, data in video_memory.items()
        ]
    }

@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    if video_id in video_memory:
        del video_memory[video_id]
        return {"status": "deleted", "video_id": video_id}
    raise HTTPException(status_code=404, detail="Video not found")