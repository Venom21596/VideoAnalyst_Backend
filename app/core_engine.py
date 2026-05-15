# app/core_engine.py


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
import cv2
import re

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Loading AI models...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
whisper_model = whisper.load_model("base")
print("Models ready.")


def has_audio(video_path: str) -> bool:
    """
    Return True only if the video contains meaningful audio.
    Silent audio tracks are treated as no audio.
    """

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", "volumedetect",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    output = result.stderr

    mean_volume = None

    for line in output.splitlines():
        if "mean_volume:" in line:
            try:
                mean_volume = float(
                    line.split("mean_volume:")[1]
                    .split(" dB")[0]
                    .strip()
                )
            except:
                pass

    if mean_volume is None:
        return False

    return mean_volume > -50


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def extract_audio(video_path: str):
    """Extract audio for faster Whisper transcription."""

    os.makedirs("data", exist_ok=True)

    audio_path = "data/temp_audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(
        cmd,
        capture_output=True
    )

    return audio_path


def extract_frames(video_path: str, interval_seconds: int = 5) -> List[Dict]:
    """Extract frames from video at regular intervals."""

    frames_dir = "data/frames"
    os.makedirs(frames_dir, exist_ok=True)

    duration = get_video_duration(video_path)

    frames = []
    timestamp = 0

    while timestamp < duration:

        frame_path = f"{frames_dir}/frame_{int(timestamp)}.jpg"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "8",
            frame_path
        ]

        subprocess.run(
            cmd,
            capture_output=True
        )

        if os.path.exists(frame_path):

            frame = cv2.imread(frame_path)

            if frame is not None:
                frame = cv2.resize(frame, (640, 360))
                cv2.imwrite(frame_path, frame)

            frames.append({
                "path": frame_path,
                "timestamp": timestamp
            })

        timestamp += interval_seconds

    return frames


def describe_frame_with_vision_model(frame_path: str) -> str:
    """Describe a frame using Qwen2.5-VL."""

    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl",
                "prompt": """
Generate a factual visual description.

Rules:
- Describe only directly visible objects and actions.
- Do not infer story, meaning, purpose, or condition.
- Do not describe emotions or intentions.
- Do not exaggerate scene details.
- Use simple factual sentences.
- Mention only clearly visible elements.
- Maximum 1 short sentence.

Example:
'A chicken is standing near a road with grass on the side.'

Do NOT generate interpretations.
""",
                "images": [image_data],
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        return response.json()["response"]

    except Exception as e:

        logger.error(
            f"Vision model error: {e}"
        )

        return "Frame could not be described."


def transcribe_silent_video(video_path: str) -> List[Dict]:
    """Generate visual transcript for silent videos."""

    logger.info(
        "No audio detected. Using multimodal visual analysis..."
    )

    duration = get_video_duration(video_path)

    interval = 2 if duration < 120 else 5

    frames = extract_frames(
        video_path,
        interval_seconds=interval
    )

    segments = []

    with ThreadPoolExecutor(max_workers=4) as executor:

        descriptions = list(
            executor.map(
                lambda frame: describe_frame_with_vision_model(frame["path"]),
                frames
            )
        )

    for frame, description in zip(frames, descriptions):

        segments.append({
            "text": description,
            "start": frame["timestamp"],
            "end": frame["timestamp"] + interval
        })

        logger.info(
            f"Frame at {frame['timestamp']}s described."
        )

    return segments


def process_video(video_path: str) -> Dict[str, Any]:

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"File not found: {video_path}"
        )

    logger.info("Processing video...")

    if has_audio(video_path):

        logger.info(
            "Audio detected. Transcribing with Whisper..."
        )

        audio_path = extract_audio(video_path)

        with ThreadPoolExecutor(max_workers=2) as executor:

            scene_future = executor.submit(
                detect,
                video_path,
                ContentDetector()
            )

            transcribe_future = executor.submit(
                whisper_model.transcribe,
                audio_path,
                fp16=False,
                verbose=False,
                condition_on_previous_text=False
            )

            scene_list = scene_future.result()

            result = transcribe_future.result()

            raw_segments = result["segments"]

            segments = []

            for seg in raw_segments:

                text = seg["text"].strip()

                sentences = re.split(
                    r'(?<=[.!?])\s+',
                    text
                )

                if not sentences:
                    continue

                seg_duration = (
                    seg["end"] - seg["start"]
                )

                sentence_duration = (
                    seg_duration / max(len(sentences), 1)
                )

                current_start = seg["start"]

                for sentence in sentences:

                    sentence = sentence.strip()

                    if not sentence:
                        continue

                    segments.append({
                        "text": sentence,
                        "start": current_start,
                        "end": current_start + sentence_duration
                    })

                    current_start += sentence_duration

    else:

        logger.info(
            "No audio. Analysing frames with multimodal vision model..."
        )

        scene_list = detect(
            video_path,
            ContentDetector()
        )

        segments = transcribe_silent_video(
            video_path
        )

    chapters = [
        scene[0].get_seconds()
        for scene in scene_list
    ]

    logger.info(
        "Building search index..."
    )

    texts = [
        s["text"]
        for s in segments
        if s.get("text")
    ]

    if not texts:
        raise ValueError(
            "No transcript or visual descriptions generated."
        )

    embeddings = embed_model.encode(
        texts
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return {
        "index": index,
        "segments": segments,
        "texts": texts,
        "chapters": chapters,
        "has_audio": has_audio(video_path)
    }


def search_video(
    query: str,
    index,
    texts: List[str],
    segments: List[Dict]
) -> str:

    if not query or not query.strip():
        return "Error: Empty question."

    if len(query) > 1000:
        return "Error: Question too long."

    timestamp_match = re.findall(
        r'(\d{1,2})[:.](\d{2})[:.](\d{2})',
        query
    )

    if timestamp_match:

        h, m, s = map(int, timestamp_match[0])

        target_seconds = h * 3600 + m * 60 + s

        nearby_segments = []

        for seg in segments:

            if abs(seg["start"] - target_seconds) <= 10:

                nearby_segments.append(seg)

        if nearby_segments:

            context = "\n".join([
                f"[{seg['start']:.1f}s] {seg['text']}"
                for seg in nearby_segments
            ])

        else:

            context = "No matching timestamp context found."

    else:

        query_vec = embed_model.encode(
            [query]
        ).astype('float32')

        D, I = index.search(
            query_vec,
            k=3
        )

        context_chunks = []

        for idx in I[0]:

            start = max(0, idx - 2)
            end = min(len(texts), idx + 3)

            window = " ".join(
                texts[start:end]
            )

            timestamp = segments[idx]['start']

            context_chunks.append(
                f"[{timestamp:.1f}s]: {window}"
            )

        context = "\n---\n".join(
            context_chunks
        )

    prompt = (
    "You are a transcript-based video assistant.\n\n"

    "Instructions:\n"
    "- Use ONLY the provided transcript context.\n"
    "- Do NOT invent events or facts.\n"
    "- Provide a detailed but grounded explanation.\n"
    "- Summarize the events clearly.\n"
    "- Mention important people, places, and topics.\n"
    "- Explain the sequence of events naturally.\n"
    "- If information is incomplete, explicitly say so.\n\n"

    f"Transcript Context:\n{context}\n\n"
    f"Question: {query}"
)

    try:

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        res.raise_for_status()

        return res.json()['response']

    except requests.exceptions.ConnectionError:

        return (
            "Error: Ollama is not running. "
            "Start it using: ollama serve"
        )

    except Exception as e:

        logger.error(
            f"LLM error: {e}"
        )

        return "Error: Something went wrong with the LLM."
