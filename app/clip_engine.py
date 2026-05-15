import subprocess
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def find_clip_boundaries(
    description: str,
    segments: List[Dict],
    embed_model
) -> Dict[str, float]:
    """
    Find start and end time for a clip
    based on natural language description.
    """

    import faiss
    import numpy as np

    texts = [s['text'] for s in segments]

    embeddings = embed_model.encode(
        texts
    ).astype('float32')

    index = faiss.IndexFlatL2(
        int(embeddings.shape[1])
    )

    index.add(embeddings)

    query_vec = embed_model.encode(
        [description]
    ).astype('float32')

    D, I = index.search(
        query_vec,
        k=3
    )  # type: ignore

    matched_indices = sorted(I[0])

    start_time = segments[
        matched_indices[0]
    ]['start']

    end_time = segments[
        matched_indices[-1]
    ]['end']

    # Add 2s padding
    start_time = max(0, start_time - 2)
    end_time = end_time + 2

    return {
        "start": start_time,
        "end": end_time
    }


def manual_clip_boundaries(
    start_time: str,
    end_time: str
) -> Dict[str, float]:
    """
    Convert HH:MM:SS timestamps
    into seconds.
    """

    def to_seconds(t: str):

        h, m, s = map(
            int,
            t.split(":")
        )

        return (
            h * 3600
            + m * 60
            + s
        )

    return {
        "start": to_seconds(start_time),
        "end": to_seconds(end_time)
    }


def export_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str
) -> str:
    """
    Use FFmpeg to trim a clip
    from the video.
    """

    duration = end - start

    if duration <= 0:
        raise ValueError(
            "Invalid clip boundaries."
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        logger.error(
            f"FFmpeg error: {result.stderr}"
        )

        raise RuntimeError(
            f"FFmpeg failed: {result.stderr}"
        )

    return output_path