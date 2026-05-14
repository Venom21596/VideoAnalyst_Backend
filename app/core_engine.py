import whisper
import faiss
import numpy as np
from scenedetect import detect, ContentDetector
from sentence_transformers import SentenceTransformer
import os
import logging
import subprocess
import requests
import base64
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Loading AI models...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = whisper.load_model("base")
print("Models ready.")

def has_audio(video_path: str) -> bool:
    """Check if video has an audio track."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "audio" in result.stdout

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def extract_frames(video_path: str, interval_seconds: int = 5) -> List[Dict]:
    """Extract frames from video at regular intervals using FFmpeg."""
    frames_dir = "data/frames"
    os.makedirs(frames_dir, exist_ok=True)

    duration = get_video_duration(video_path)
    frames = []
    timestamp = 0

    while timestamp < duration:
        frame_path = f"{frames_dir}/frame_{int(timestamp)}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            frame_path
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(frame_path):
            frames.append({"path": frame_path, "timestamp": timestamp})
        timestamp += interval_seconds

    return frames

def describe_frame_with_llava(frame_path: str) -> str:
    """Use LLaVA to describe what's happening in a video frame."""
    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "Describe what is happening in this video frame in 1-2 sentences. Focus on actions, text visible, people, objects, and any important visual information.",
                "images": [image_data],
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        logger.error(f"LLaVA error: {e}")
        return "Frame could not be described."

def transcribe_silent_video(video_path: str) -> List[Dict]:
    """For silent videos — extract frames and describe each with LLaVA."""
    logger.info("No audio detected. Using LLaVA visual analysis...")
    frames = extract_frames(video_path, interval_seconds=5)
    segments = []

    for frame in frames:
        description = describe_frame_with_llava(frame["path"])
        segments.append({
            "text": description,
            "start": frame["timestamp"],
            "end": frame["timestamp"] + 5
        })
        logger.info(f"Frame at {frame['timestamp']}s described.")

    return segments

def process_video(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    # Scene detection (works for all videos)
    logger.info("Detecting scenes...")
    scene_list = detect(video_path, ContentDetector())
    chapters = [scene[0].get_seconds() for scene in scene_list]

    # Check for audio
    if has_audio(video_path):
        logger.info("Audio detected. Transcribing with Whisper...")
        result = whisper_model.transcribe(video_path)
        segments = result['segments']
    else:
        logger.info("No audio. Analysing frames with LLaVA...")
        segments = transcribe_silent_video(video_path)

    # Build FAISS index
    logger.info("Building search index...")
    texts = [s['text'] for s in segments]
    embeddings = embed_model.encode(texts).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return {
        "index": index,
        "segments": segments,
        "texts": texts,
        "chapters": chapters,
        "has_audio": has_audio(video_path)
    }

def search_video(query: str, index, texts: List[str], segments: List[Dict]) -> str:
    if not query or not query.strip():
        return "Error: Empty question."
    if len(query) > 1000:
        return "Error: Question too long."

    query_vec = embed_model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k=3)

    context_chunks = []
    for idx in I[0]:
        start = max(0, idx - 2)
        end = min(len(texts), idx + 3)
        window = " ".join(texts[start:end])
        timestamp = segments[idx]['start']
        context_chunks.append(f"[{timestamp:.1f}s]: {window}")

    context = "\n---\n".join(context_chunks)
    prompt = (
        f"You are a Video Analyst. Answer using only the context below. "
        f"Always mention timestamps.\n\nContext:\n{context}\n\nQuestion: {query}"
    )

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
            timeout=60
        )
        res.raise_for_status()
        return res.json()['response']
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with: ollama run llama3.2:1b"
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Error: Something went wrong with the LLM."