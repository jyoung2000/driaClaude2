# DriaClaude2 - Advanced TTS with Voice Cloning

A Docker container that provides a web GUI and API for the Dia TTS model with advanced voice cloning, granular control, and batch processing capabilities. Designed to run on unRAID and other Docker environments.

## Features

- 🎙️ **Ultra-realistic TTS** - Generate natural dialogue with multiple speakers
- 🎭 **Voice Cloning** - Clone and save voices from audio samples
- 🎛️ **Granular Control** - Fine-tune generation parameters (temperature, top-p, top-k, etc.)
- 🎵 **Nonverbal Sounds** - Support for laughs, sighs, coughs, and other expressions
- 📦 **Batch Processing** - Generate multiple audio files at once
- 🌐 **Web GUI** - User-friendly Gradio interface on port 4144
- 🔌 **REST API** - Full-featured API for integration with other services
- 💾 **Persistent Storage** - Save voices and outputs across container restarts
- 🐳 **unRAID Compatible** - Proper permissions and configuration for unRAID

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/jyoung2000/driaClaude2.git
cd driaClaude2
```

2. Build and run:
```bash
docker-compose build
docker-compose up -d
```

3. Access the web interface at `http://localhost:4144/ui`

### Using Docker CLI

```bash
docker build -t driaclaude2 .
docker run -d \
  --name driaclaude2 \
  -p 4144:4144 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/models:/home/appuser/.cache/huggingface \
  -e PUID=99 \
  -e PGID=100 \
  driaclaude2
```

## Web Interface

Access the Gradio interface at `http://localhost:4144/ui`

### Features:

1. **Single Generation Tab**
   - Text input with speaker tags `[S1]` and `[S2]`
   - Voice selection for cloning
   - Advanced settings (guidance scale, temperature, etc.)
   - Nonverbal sound options
   - Audio preview and download

2. **Voice Cloning Tab**
   - Upload 5-10 second audio samples
   - Provide accurate transcripts
   - Tag and organize voices
   - Manage cloned voices

3. **Batch Processing Tab**
   - Generate multiple audio files
   - JSON input format
   - Zip download option

## API Documentation

Full API documentation available at `http://localhost:4144/docs`

### Key Endpoints:

- `POST /api/tts` - Generate single audio file
- `POST /api/tts/batch` - Batch generation
- `POST /api/voice/clone` - Clone a voice
- `GET /api/voices` - List cloned voices
- `GET /api/outputs` - List generated files
- `GET /api/download/{filename}` - Download audio file

### Example API Usage:

```python
import requests
import json

# Generate TTS
response = requests.post(
    "http://localhost:4144/api/tts",
    json={
        "text": "[S1] Hello world! [S2] This is amazing!",
        "voice_id": None,
        "guidance_scale": 3.0,
        "temperature": 1.8,
        "output_format": "mp3"
    }
)

# Save the audio
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## Voice Cloning Guidelines

For best results when cloning voices:

1. Use 5-10 seconds of clear audio
2. Provide accurate transcripts with proper speaker tags
3. Use high-quality recordings without background noise
4. Single speaker audio works best
5. Include the exact words spoken in the transcript

## Text Format Guidelines

- Always start with `[S1]` for the first speaker
- Alternate between `[S1]` and `[S2]` for dialogue
- Use `(laughs)`, `(sighs)`, etc. for nonverbal sounds
- Keep individual segments under 20 seconds for natural pacing

Example:
```
[S1] Welcome to our podcast! (laughs) [S2] Thanks for having me. [S1] Let's dive right in.
```

## Configuration

### Environment Variables:

- `PORT` - Web server port (default: 4144)
- `PUID` - User ID for file permissions (default: 99 for unRAID)
- `PGID` - Group ID for file permissions (default: 100 for unRAID)
- `TZ` - Timezone (default: UTC)

### Volumes:

- `/app/uploads` - Temporary upload storage
- `/app/outputs` - Generated audio files
- `/app/voices` - Cloned voice data
- `/home/appuser/.cache/huggingface` - Model cache

## unRAID Installation

1. Go to Docker tab in unRAID web interface
2. Click "Add Container"
3. Fill in the following:
   - Name: `driaClaude2`
   - Repository: `driaclaude2` (after building)
   - Network Type: `bridge`
   - Port: `4144:4144`
   - Add paths for volumes as needed

## Troubleshooting

### Model Loading Issues
- First run downloads ~6GB model data
- Ensure sufficient disk space in model cache volume
- Check container logs: `docker logs driaclaude2`

### Permission Issues
- Verify PUID/PGID match your system
- Check volume mount permissions
- For unRAID, default 99:100 should work

### GPU Support
- Currently CPU-only for maximum compatibility
- GPU support may be added in future versions

## Development

### Building from Source:

```bash
# Clone repository
git clone https://github.com/jyoung2000/driaClaude2.git
cd driaClaude2

# Install dependencies (for local development)
pip install -r requirements.txt

# Run locally
python app_enhanced.py
```

### Running Tests:

```bash
# API tests
pytest tests/

# Generate test audio
python tests/test_generation.py
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built on the [Dia TTS model](https://github.com/nari-labs/dia) by Nari Labs
- Uses Hugging Face Transformers for model loading
- Gradio for the web interface
- FastAPI for the REST API

## Support

- GitHub Issues: [https://github.com/jyoung2000/driaClaude2/issues](https://github.com/jyoung2000/driaClaude2/issues)
- Documentation: [https://github.com/jyoung2000/driaClaude2/wiki](https://github.com/jyoung2000/driaClaude2/wiki)