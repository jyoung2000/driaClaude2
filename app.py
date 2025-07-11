import os
import io
import base64
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from datetime import datetime
import json
import uuid
from pathlib import Path
from transformers import AutoProcessor, DiaForConditionalGeneration
import gradio as gr

# Configuration
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
VOICES_DIR = Path("/app/voices")
MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, VOICES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="DriaClaude2 TTS API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
processor = None
model = None
device = None

class TTSRequest(BaseModel):
    text: str
    speaker_tags: Optional[List[str]] = ["[S1]", "[S2]"]
    guidance_scale: Optional[float] = 3.0
    temperature: Optional[float] = 1.8
    top_p: Optional[float] = 0.90
    top_k: Optional[int] = 45
    max_new_tokens: Optional[int] = 3072
    voice_id: Optional[str] = None
    seed: Optional[int] = None
    include_nonverbal: Optional[bool] = True
    nonverbal_tags: Optional[List[str]] = []

class VoiceCloneRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    transcript: str

def load_model():
    global processor, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = DiaForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT).to(device)
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text"""
    try:
        # Set seed if provided
        if request.seed:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)
        
        # Prepare text with speaker tags
        text = request.text
        
        # Add nonverbal tags if requested
        if request.include_nonverbal and request.nonverbal_tags:
            for tag in request.nonverbal_tags:
                text = text.replace(f"[{tag}]", f"({tag})")
        
        # Load voice prompt if voice_id is provided
        audio_prompt = None
        if request.voice_id:
            voice_path = VOICES_DIR / f"{request.voice_id}.json"
            if voice_path.exists():
                with open(voice_path, 'r') as f:
                    voice_data = json.load(f)
                    # Prepend voice transcript to text
                    text = voice_data['transcript'] + " " + text
        
        # Process input
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.guidance_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        # Decode and save audio
        audio_outputs = processor.batch_decode(outputs)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = OUTPUT_DIR / filename
        
        processor.save_audio(audio_outputs, str(output_path))
        
        # Return audio file
        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    request: str = Form(...)
):
    """Clone a voice from audio file"""
    try:
        request_data = json.loads(request)
        voice_request = VoiceCloneRequest(**request_data)
        
        # Save uploaded audio
        audio_content = await audio.read()
        temp_audio_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.wav"
        
        with open(temp_audio_path, "wb") as f:
            f.write(audio_content)
        
        # Generate voice ID
        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        
        # Save voice data
        voice_data = {
            "id": voice_id,
            "name": voice_request.name,
            "description": voice_request.description,
            "transcript": voice_request.transcript,
            "created_at": datetime.now().isoformat(),
            "audio_file": str(temp_audio_path)
        }
        
        voice_path = VOICES_DIR / f"{voice_id}.json"
        with open(voice_path, 'w') as f:
            json.dump(voice_data, f)
        
        return {"voice_id": voice_id, "message": "Voice cloned successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def list_voices():
    """List all available voices"""
    voices = []
    for voice_file in VOICES_DIR.glob("*.json"):
        with open(voice_file, 'r') as f:
            voice_data = json.load(f)
            voices.append({
                "id": voice_data["id"],
                "name": voice_data["name"],
                "description": voice_data.get("description", ""),
                "created_at": voice_data.get("created_at", "")
            })
    return {"voices": voices}

@app.delete("/api/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice"""
    voice_path = VOICES_DIR / f"{voice_id}.json"
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    
    # Load voice data to get audio file path
    with open(voice_path, 'r') as f:
        voice_data = json.load(f)
    
    # Delete audio file if exists
    if "audio_file" in voice_data:
        audio_path = Path(voice_data["audio_file"])
        if audio_path.exists():
            audio_path.unlink()
    
    # Delete voice data
    voice_path.unlink()
    
    return {"message": "Voice deleted successfully"}

@app.get("/api/outputs")
async def list_outputs():
    """List all generated audio files"""
    outputs = []
    for audio_file in OUTPUT_DIR.glob("*.mp3"):
        outputs.append({
            "filename": audio_file.name,
            "size": audio_file.stat().st_size,
            "created_at": datetime.fromtimestamp(audio_file.stat().st_mtime).isoformat()
        })
    return {"outputs": sorted(outputs, key=lambda x: x["created_at"], reverse=True)}

@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Download a generated audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Gradio interface
def create_gradio_interface():
    def generate_audio(text, voice_id, guidance_scale, temperature, top_p, top_k, seed, include_nonverbal):
        request = TTSRequest(
            text=text,
            voice_id=voice_id if voice_id != "None" else None,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed if seed != -1 else None,
            include_nonverbal=include_nonverbal
        )
        
        # Set seed if provided
        if request.seed:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)
        
        # Prepare text
        text = request.text
        
        # Load voice prompt if voice_id is provided
        if request.voice_id:
            voice_path = VOICES_DIR / f"{request.voice_id}.json"
            if voice_path.exists():
                with open(voice_path, 'r') as f:
                    voice_data = json.load(f)
                    text = voice_data['transcript'] + " " + text
        
        # Process and generate
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                guidance_scale=request.guidance_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        
        audio_outputs = processor.batch_decode(outputs)
        
        # Save to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = OUTPUT_DIR / filename
        processor.save_audio(audio_outputs, str(output_path))
        
        return str(output_path)
    
    def get_voice_choices():
        voices = ["None"]
        for voice_file in VOICES_DIR.glob("*.json"):
            with open(voice_file, 'r') as f:
                voice_data = json.load(f)
                voices.append(f"{voice_data['id']} - {voice_data['name']}")
        return voices
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=generate_audio,
        inputs=[
            gr.Textbox(
                label="Text to synthesize",
                placeholder="[S1] Hello, this is speaker one. [S2] And this is speaker two!",
                lines=3
            ),
            gr.Dropdown(
                label="Voice (for cloning)",
                choices=get_voice_choices(),
                value="None"
            ),
            gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1),
            gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.8, step=0.1),
            gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.90, step=0.05),
            gr.Slider(label="Top-k", minimum=1, maximum=100, value=45, step=1),
            gr.Number(label="Seed (-1 for random)", value=-1, precision=0),
            gr.Checkbox(label="Include nonverbal sounds", value=True)
        ],
        outputs=gr.Audio(label="Generated Audio", type="filepath"),
        title="DriaClaude2 TTS",
        description="Generate ultra-realistic dialogue with Dia TTS model",
        theme="huggingface"
    )
    
    return iface

# Mount Gradio app
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4144))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)