import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import librosa
import soundfile as sf

class VoiceCloneManager:
    """Manages voice cloning, storage, and retrieval"""
    
    def __init__(self, voices_dir: Path, processor=None):
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.processor = processor
        
    def extract_voice_features(self, audio_path: str, transcript: str) -> Dict[str, Any]:
        """Extract voice features from audio file"""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (common for speech models)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Calculate duration
        duration = waveform.shape[1] / sample_rate
        
        # Extract acoustic features
        features = {
            "duration": duration,
            "sample_rate": sample_rate,
            "transcript": transcript,
            "audio_stats": {
                "mean": float(waveform.mean()),
                "std": float(waveform.std()),
                "max": float(waveform.max()),
                "min": float(waveform.min())
            }
        }
        
        # Extract pitch information using librosa
        audio_np = waveform.squeeze().numpy()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        
        # Filter out unvoiced frames
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) > 0:
            features["pitch_stats"] = {
                "mean": float(np.mean(f0_voiced)),
                "std": float(np.std(f0_voiced)),
                "min": float(np.min(f0_voiced)),
                "max": float(np.max(f0_voiced))
            }
        
        return features
    
    def save_voice(self, 
                   audio_path: str, 
                   name: str, 
                   transcript: str,
                   description: Optional[str] = "",
                   tags: Optional[list] = None) -> str:
        """Save a voice clone with metadata"""
        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        
        # Extract features
        features = self.extract_voice_features(audio_path, transcript)
        
        # Copy audio file to voices directory
        audio_ext = Path(audio_path).suffix
        new_audio_path = self.voices_dir / f"{voice_id}{audio_ext}"
        
        # Load and save audio with normalization
        waveform, sample_rate = torchaudio.load(audio_path)
        # Normalize audio
        waveform = waveform / waveform.abs().max()
        torchaudio.save(str(new_audio_path), waveform, sample_rate)
        
        # Create metadata
        metadata = {
            "id": voice_id,
            "name": name,
            "description": description,
            "transcript": transcript,
            "tags": tags or [],
            "features": features,
            "audio_file": str(new_audio_path),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = self.voices_dir / f"{voice_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return voice_id
    
    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve voice metadata"""
        metadata_path = self.voices_dir / f"{voice_id}.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_voices(self, tags: Optional[list] = None) -> list:
        """List all available voices, optionally filtered by tags"""
        voices = []
        for metadata_file in self.voices_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                voice_data = json.load(f)
                
                # Filter by tags if provided
                if tags:
                    voice_tags = set(voice_data.get("tags", []))
                    if not any(tag in voice_tags for tag in tags):
                        continue
                
                voices.append({
                    "id": voice_data["id"],
                    "name": voice_data["name"],
                    "description": voice_data.get("description", ""),
                    "tags": voice_data.get("tags", []),
                    "duration": voice_data.get("features", {}).get("duration", 0),
                    "created_at": voice_data.get("created_at", "")
                })
        
        return sorted(voices, key=lambda x: x["created_at"], reverse=True)
    
    def update_voice(self, voice_id: str, **kwargs) -> bool:
        """Update voice metadata"""
        voice_data = self.get_voice(voice_id)
        if not voice_data:
            return False
        
        # Update allowed fields
        allowed_fields = ["name", "description", "tags"]
        for field in allowed_fields:
            if field in kwargs:
                voice_data[field] = kwargs[field]
        
        voice_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated metadata
        metadata_path = self.voices_dir / f"{voice_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(voice_data, f, indent=2)
        
        return True
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice and its associated files"""
        voice_data = self.get_voice(voice_id)
        if not voice_data:
            return False
        
        # Delete audio file
        audio_path = Path(voice_data["audio_file"])
        if audio_path.exists():
            audio_path.unlink()
        
        # Delete metadata
        metadata_path = self.voices_dir / f"{voice_id}.json"
        metadata_path.unlink()
        
        return True
    
    def prepare_voice_prompt(self, voice_id: str) -> Optional[str]:
        """Prepare voice prompt text for TTS generation"""
        voice_data = self.get_voice(voice_id)
        if not voice_data:
            return None
        
        return voice_data["transcript"]