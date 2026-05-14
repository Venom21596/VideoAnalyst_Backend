import cv2
import os
import requests
from typing import List, Dict

FRAME_DIR = "data/frames"

def extract_frames(video_path: str, interval=5) -> List[Dict]:
    """
    Extract one frame every N seconds
    """

    os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frames=[]

    count=0
    timestamp=0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % frame_interval == 0:

            path=f"{FRAME_DIR}/frame_{count}.jpg"

            cv2.imwrite(path, frame)

            frames.append({
                "path":path,
                "timestamp":timestamp
            })

            timestamp += interval

        count+=1

    cap.release()

    return frames


def generate_visual_segments(frames):

    visual_segments=[]

    for frame in frames:

        try:

            response=requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":"llava",
                    "prompt":"Describe what is happening in this image in one sentence.",
                    "images":[frame["path"]],
                    "stream":False
                },
                timeout=120
            )

            description=response.json()["response"] 

            visual_segments.append({
                "start":frame["timestamp"],
                "end":frame["timestamp"]+5,
                "text":description
            })

        except Exception as e:

            print(e)

    return visual_segments