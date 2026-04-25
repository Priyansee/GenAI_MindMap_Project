from __future__ import annotations
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from pipeline_controller import PipelineController
from agents.clarifier_agent import ClarifierAgent
import os
from dotenv import load_dotenv

# Explicitly load .env from the same directory as this file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = FastAPI(title="MindForge API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    text: str
    clarification_answers: Optional[Dict[str, str]] = None
    additional_info: Optional[str] = None
    api_key: Optional[str] = None

class RefineRequest(BaseModel):
    previous_map: Dict
    feedback: str
    api_key: Optional[str] = None

@app.post("/clarify")
async def get_clarification_questions(request: GenerateRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    # Controller Agent Logic: Check if input is vague
    word_count = len(request.text.split())
    is_vague = word_count < 5
    
    api_key = request.api_key or os.getenv("GROK_API_KEY")
    clarifier = ClarifierAgent(api_key=api_key)
    try:
        result = await clarifier.run({"input": request.text})
        # Wrap in a standard response with is_vague flag
        return {
            "is_vague": is_vague,
            "word_count": word_count,
            "questions": result.get("questions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clarification failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Welcome to MindForge API",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/generate")
async def generate_mind_map(request: GenerateRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    # Use API key from request or environment
    api_key = request.api_key or os.getenv("GROK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required (pass in request or set GROK_API_KEY env var)")

    controller = PipelineController(api_key=api_key)
    try:
        result = await controller.run(
            request.text, 
            clarification_answers=request.clarification_answers,
            additional_info=request.additional_info
        )
        return result
    except Exception as e:
        import traceback
        print(f"PIPELINE ERROR: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/refine")
async def refine_mind_map(request: RefineRequest):
    if not request.feedback:
        raise HTTPException(status_code=400, detail="Feedback is required")
    
    api_key = request.api_key or os.getenv("GROK_API_KEY")
    controller = PipelineController(api_key=api_key)
    try:
        result = await controller.refine(request.previous_map, request.feedback)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
