import os
import io
import base64
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import json
import uuid
from pathlib import Path
from transformers import AutoProcessor, DiaForConditionalGeneration
import gradio as gr
import asyncio
from voice_cloning import VoiceCloneManager
import zipfile
import tempfile

# Configuration
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
VOICES_DIR = Path("/app/voices")
MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, VOICES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="DriaClaude2 TTS API",
    version="2.0.0",
    description="Advanced TTS with voice cloning, granular control, and batch processing"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor = None
model = None
device = None
voice_manager = None

# Supported nonverbal tags
NONVERBAL_TAGS = [
    "laughs", "clears throat", "sighs", "gasps", "coughs", "singing", 
    "sings", "mumbles", "beep", "groans", "sniffs", "claps", "screams",
    "inhales", "exhales", "applause", "burps", "humming", "sneezes",
    "chuckle", "whistles"
]

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize with speaker tags [S1], [S2]")
    voice_id: Optional[str] = Field(None, description="Voice ID for cloning")
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0)
    temperature: float = Field(1.8, ge=0.1, le=2.0)
    top_p: float = Field(0.90, ge=0.1, le=1.0)
    top_k: int = Field(45, ge=1, le=100)
    max_new_tokens: int = Field(3072, ge=100, le=10000)
    seed: Optional[int] = None
    include_nonverbal: bool = True
    nonverbal_tags: List[str] = Field(default_factory=list)
    output_format: str = Field("mp3", pattern="^(mp3|wav)$")

class BatchTTSRequest(BaseModel):
    requests: List[TTSRequest]
    zip_output: bool = True

class VoiceCloneRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    transcript: str
    tags: Optional[List[str]] = Field(default_factory=list)

class VoiceUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

def load_model():
    global processor, model, device, voice_manager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = DiaForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT).to(device)
    voice_manager = VoiceCloneManager(VOICES_DIR, processor)
    
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DriaClaude2 TTS API",
        "version": "2.0.0",
        "endpoints": {
            "webui": "/ui",
            "api_docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }

def process_text_with_nonverbal(text: str, nonverbal_tags: List[str]) -> str:
    """Process text to include nonverbal tags"""
    for tag in nonverbal_tags:
        if tag in NONVERBAL_TAGS:
            # Replace [tag] with (tag) format
            text = text.replace(f"[{tag}]", f"({tag})")
    return text

async def generate_audio(request: TTSRequest) -> Path:
    """Generate audio from TTS request"""
    # Set seed if provided
    if request.seed:
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)
    
    # Prepare text
    text = request.text
    
    # Add nonverbal tags if requested
    if request.include_nonverbal:
        text = process_text_with_nonverbal(text, request.nonverbal_tags)
    
    # Load voice prompt if voice_id is provided
    if request.voice_id:
        voice_prompt = voice_manager.prepare_voice_prompt(request.voice_id)
        if voice_prompt:
            text = voice_prompt + " " + text
    
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
    filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.{request.output_format}"
    output_path = OUTPUT_DIR / filename
    
    processor.save_audio(audio_outputs, str(output_path))
    
    return output_path

@app.post("/api/tts", response_model=None)
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text"""
    try:
        output_path = await generate_audio(request)
        
        return FileResponse(
            path=output_path,
            media_type=f"audio/{request.output_format}",
            filename=output_path.name,
            headers={"Content-Disposition": f"attachment; filename={output_path.name}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/batch")
async def generate_batch_tts(request: BatchTTSRequest, background_tasks: BackgroundTasks):
    """Generate multiple TTS audio files in batch"""
    try:
        output_files = []
        
        for tts_request in request.requests:
            output_path = await generate_audio(tts_request)
            output_files.append(output_path)
        
        if request.zip_output:
            # Create zip file
            zip_path = OUTPUT_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in output_files:
                    zipf.write(file_path, file_path.name)
            
            # Clean up individual files after zipping
            background_tasks.add_task(cleanup_files, output_files)
            
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename=zip_path.name
            )
        else:
            return {
                "files": [str(f.name) for f in output_files],
                "message": f"Generated {len(output_files)} audio files"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_files(files: List[Path]):
    """Clean up temporary files"""
    for file_path in files:
        if file_path.exists():
            file_path.unlink()

@app.post("/api/voice/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    data: str = Form(...)
):
    """Clone a voice from audio file"""
    try:
        request_data = json.loads(data)
        voice_request = VoiceCloneRequest(**request_data)
        
        # Save uploaded audio temporarily
        audio_content = await audio.read()
        temp_audio_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.wav"
        
        with open(temp_audio_path, "wb") as f:
            f.write(audio_content)
        
        # Save voice using voice manager
        voice_id = voice_manager.save_voice(
            audio_path=str(temp_audio_path),
            name=voice_request.name,
            transcript=voice_request.transcript,
            description=voice_request.description,
            tags=voice_request.tags
        )
        
        # Clean up temp file
        temp_audio_path.unlink()
        
        return {
            "voice_id": voice_id,
            "message": "Voice cloned successfully",
            "details": voice_manager.get_voice(voice_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def list_voices(tags: Optional[str] = None):
    """List all available voices"""
    tag_list = tags.split(",") if tags else None
    voices = voice_manager.list_voices(tags=tag_list)
    return {"voices": voices, "total": len(voices)}

@app.get("/api/voice/{voice_id}")
async def get_voice(voice_id: str):
    """Get detailed information about a voice"""
    voice_data = voice_manager.get_voice(voice_id)
    if not voice_data:
        raise HTTPException(status_code=404, detail="Voice not found")
    return voice_data

@app.patch("/api/voice/{voice_id}")
async def update_voice(voice_id: str, update_request: VoiceUpdateRequest):
    """Update voice metadata"""
    update_data = update_request.dict(exclude_unset=True)
    if voice_manager.update_voice(voice_id, **update_data):
        return {"message": "Voice updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Voice not found")

@app.delete("/api/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice"""
    if voice_manager.delete_voice(voice_id):
        return {"message": "Voice deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Voice not found")

@app.get("/api/outputs")
async def list_outputs(limit: int = 50):
    """List generated audio files"""
    outputs = []
    audio_files = list(OUTPUT_DIR.glob("*.mp3")) + list(OUTPUT_DIR.glob("*.wav"))
    
    for audio_file in sorted(audio_files, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        outputs.append({
            "filename": audio_file.name,
            "size": audio_file.stat().st_size,
            "format": audio_file.suffix[1:],
            "created_at": datetime.fromtimestamp(audio_file.stat().st_mtime).isoformat()
        })
    
    return {"outputs": outputs, "total": len(outputs)}

@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Download a generated audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type=f"audio/{file_path.suffix[1:]}",
        filename=filename
    )

@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """Delete a generated audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path.unlink()
    return {"message": "File deleted successfully"}

@app.get("/api/nonverbal-tags")
async def get_nonverbal_tags():
    """Get list of supported nonverbal tags"""
    return {"tags": NONVERBAL_TAGS}

# Create enhanced Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="DriaClaude2 TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DriaClaude2 TTS - Ultra-Realistic Speech Synthesis")
        gr.Markdown("Generate natural dialogue with multiple speakers, voice cloning, and fine-grained control.")
        
        with gr.Tabs():
            # Single Generation Tab
            with gr.TabItem("Single Generation"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="[S1] Hello, this is speaker one. [S2] And this is speaker two! [S1] (laughs) That's amazing!",
                            lines=5
                        )
                        
                        with gr.Row():
                            voice_dropdown = gr.Dropdown(
                                label="Voice (for cloning)",
                                choices=["None"] + [f"{v['id']} - {v['name']}" for v in voice_manager.list_voices()],
                                value="None"
                            )
                            refresh_voices_btn = gr.Button("🔄", scale=0)
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1)
                            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.8, step=0.1)
                            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.90, step=0.05)
                            top_k = gr.Slider(label="Top-k", minimum=1, maximum=100, value=45, step=1)
                            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                            output_format = gr.Radio(["mp3", "wav"], label="Output Format", value="mp3")
                        
                        with gr.Row():
                            include_nonverbal = gr.Checkbox(label="Enable nonverbal sounds", value=True)
                            nonverbal_selector = gr.CheckboxGroup(
                                choices=NONVERBAL_TAGS[:10],  # Show first 10
                                label="Common nonverbal tags",
                                visible=True
                            )
                        
                        generate_btn = gr.Button("Generate Audio", variant="primary")
                    
                    with gr.Column():
                        audio_output = gr.Audio(label="Generated Audio", type="filepath")
                        download_btn = gr.Button("Download All Outputs", variant="secondary")
                        status_text = gr.Textbox(label="Status", interactive=False)
            
            # Voice Cloning Tab
            with gr.TabItem("Voice Cloning"):
                with gr.Row():
                    with gr.Column():
                        clone_audio = gr.Audio(label="Upload voice sample (5-10 seconds)", type="filepath")
                        clone_name = gr.Textbox(label="Voice name", placeholder="John Doe")
                        clone_transcript = gr.Textbox(
                            label="Transcript of audio",
                            placeholder="[S1] This is exactly what I'm saying in the audio file.",
                            lines=3
                        )
                        clone_description = gr.Textbox(
                            label="Description (optional)",
                            placeholder="Deep male voice with American accent"
                        )
                        clone_tags = gr.Textbox(
                            label="Tags (comma-separated)",
                            placeholder="male, deep, american"
                        )
                        clone_btn = gr.Button("Clone Voice", variant="primary")
                    
                    with gr.Column():
                        clone_status = gr.Textbox(label="Clone Status", interactive=False)
                        voices_list = gr.Dataframe(
                            headers=["ID", "Name", "Tags", "Created"],
                            label="Cloned Voices"
                        )
                        delete_voice_btn = gr.Button("Delete Selected Voice", variant="stop")
            
            # Batch Processing Tab
            with gr.TabItem("Batch Processing"):
                gr.Markdown("### Generate multiple audio files at once")
                batch_input = gr.Textbox(
                    label="Batch Input (JSON format)",
                    placeholder='[\n  {"text": "[S1] First text", "voice_id": null},\n  {"text": "[S2] Second text", "voice_id": "voice_123"}\n]',
                    lines=10
                )
                batch_settings = gr.Checkbox(label="Use same settings for all", value=True)
                batch_generate_btn = gr.Button("Generate Batch", variant="primary")
                batch_output = gr.File(label="Download Batch Results")
        
        # Event handlers
        def refresh_voices():
            voices = ["None"] + [f"{v['id']} - {v['name']}" for v in voice_manager.list_voices()]
            return gr.update(choices=voices)
        
        def generate_single(text, voice_id, guidance, temp, top_p, top_k, seed, format, nonverbal, tags):
            try:
                voice_id = voice_id.split(" - ")[0] if voice_id != "None" else None
                
                request = TTSRequest(
                    text=text,
                    voice_id=voice_id,
                    guidance_scale=guidance,
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed if seed != -1 else None,
                    output_format=format,
                    include_nonverbal=nonverbal,
                    nonverbal_tags=tags
                )
                
                # Generate audio synchronously for Gradio
                output_path = asyncio.run(generate_audio(request))
                
                return str(output_path), "✅ Audio generated successfully!"
            except Exception as e:
                return None, f"❌ Error: {str(e)}"
        
        def clone_voice_handler(audio, name, transcript, description, tags):
            try:
                if not audio or not name or not transcript:
                    return "❌ Please provide audio file, name, and transcript", None
                
                tags_list = [t.strip() for t in tags.split(",")] if tags else []
                
                # Read audio file
                with open(audio, 'rb') as f:
                    audio_content = f.read()
                
                # Save temporary file
                temp_path = UPLOAD_DIR / f"temp_clone_{uuid.uuid4().hex}.wav"
                with open(temp_path, 'wb') as f:
                    f.write(audio_content)
                
                # Clone voice
                voice_id = voice_manager.save_voice(
                    audio_path=str(temp_path),
                    name=name,
                    transcript=transcript,
                    description=description,
                    tags=tags_list
                )
                
                temp_path.unlink()
                
                # Update voices list
                voices_data = [[v['id'], v['name'], ", ".join(v['tags']), v['created_at']] 
                              for v in voice_manager.list_voices()]
                
                return f"✅ Voice cloned successfully! ID: {voice_id}", voices_data
            except Exception as e:
                return f"❌ Error: {str(e)}", None
        
        # Connect event handlers
        refresh_voices_btn.click(refresh_voices, outputs=voice_dropdown)
        generate_btn.click(
            generate_single,
            inputs=[text_input, voice_dropdown, guidance_scale, temperature, 
                   top_p, top_k, seed, output_format, include_nonverbal, nonverbal_selector],
            outputs=[audio_output, status_text]
        )
        clone_btn.click(
            clone_voice_handler,
            inputs=[clone_audio, clone_name, clone_transcript, clone_description, clone_tags],
            outputs=[clone_status, voices_list]
        )
        
        # Load initial voices
        demo.load(
            lambda: [[v['id'], v['name'], ", ".join(v['tags']), v['created_at']] 
                    for v in voice_manager.list_voices()],
            outputs=voices_list
        )
    
    return demo

# Mount Gradio app
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4144))
    uvicorn.run("app_enhanced:app", host="0.0.0.0", port=port, reload=False)