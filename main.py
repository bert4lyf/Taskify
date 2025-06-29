from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from granite2 import run  
from fastapi.responses import FileResponse


app = FastAPI()

# Allow CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing: allow any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return FileResponse("/main3.html") 

@app.post("/process-transcript")
async def process_transcript(request: Request):
    data = await request.json()
    transcript = data.get("transcript")
    if not transcript:
        return {"error": "Transcript not provided."}
    result = await run(transcript)
    return {"result": result}
