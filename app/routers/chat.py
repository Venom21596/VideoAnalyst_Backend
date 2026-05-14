from fastapi import APIRouter, HTTPException
from app.core_engine import search_video
from app.state import video_memory

router = APIRouter()

@router.get("/chat")
async def chat(question: str, video_id: str | None = None):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long.")

    target = video_id
    if not target:
        if not video_memory:
            raise HTTPException(status_code=400, detail="No video processed yet.")
        target = max(video_memory.keys(), key=lambda k: video_memory[k].get('timestamp', 0))

    mem = video_memory.get(target)
    if not mem:
        raise HTTPException(status_code=404, detail=f"Video '{target}' not found.")

    answer = search_video(question, mem["index"], mem["texts"], mem["segments"])
    return {"answer": answer, "video_id": target}