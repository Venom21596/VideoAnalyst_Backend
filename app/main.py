from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import video, chat, clips

app = FastAPI(title="Video Analyst AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video.router, tags=["Video"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(clips.router, tags=["Clips"])

@app.get("/")
def root():
    return {"message": "Video Analyst API is running!"}