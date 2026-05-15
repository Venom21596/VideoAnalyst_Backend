from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.clip_engine import (
    find_clip_boundaries,
    export_clip,
    manual_clip_boundaries
)

from app.state import video_memory, embed_model

from pydantic import BaseModel

import os
import time

router = APIRouter()


class ClipRequest(BaseModel):

    video_id: str

    description: str | None = None

    start_time: str | None = None

    end_time: str | None = None


@router.post("/clip")
async def create_clip(req: ClipRequest):

    mem = video_memory.get(req.video_id)

    if not mem:

        raise HTTPException(
            status_code=404,
            detail="Video not found."
        )

    try:

        # TIMESTAMP MODE
        if req.start_time and req.end_time:

            boundaries = manual_clip_boundaries(
                req.start_time,
                req.end_time
            )

        # SEMANTIC SEARCH MODE
        else:

            if not req.description:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provide either description "
                        "or start_time/end_time."
                    )
                )

            boundaries = find_clip_boundaries(
                req.description,
                mem["segments"],
                embed_model
            )

        os.makedirs(
            "data/clips",
            exist_ok=True
        )

        output_path = (
            f"data/clips/"
            f"clip_{int(time.time())}.mp4"
        )

        export_clip(
            mem["file_path"],
            boundaries["start"],
            boundaries["end"],
            output_path
        )

        return {
            "status": "success",
            "start": boundaries["start"],
            "end": boundaries["end"],
            "clip_path": output_path
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/clip/download/{filename}")
async def download_clip(filename: str):

    path = f"data/clips/{filename}"

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Clip not found."
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
    )