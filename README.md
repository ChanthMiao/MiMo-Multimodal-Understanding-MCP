# MiMo Multimodal Understanding MCP Server

MCP server for Xiaomi MiMo multimodal understanding API (image, audio, video).

## Features

- **Image Understanding**: Single/multiple images, URL and local file support
- **Audio Understanding**: Single/multiple audio, URL and local file support
- **Video Understanding**: Single/multiple video, URL and local file support, configurable fps and resolution

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure API Key

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Or set environment variable directly:

```bash
export MIMO_API_KEY=your_api_key_here
```

Get your API key from: https://platform.xiaomimimo.com

## Usage

### Development mode (with MCP Inspector)

```bash
uv run mcp dev src/mimo_multimodal_mcp/server.py
```

### Install to Claude Desktop

```bash
uv run mcp install src/mimo_multimodal_mcp/server.py
```

### Direct execution

```bash
uv run python src/mimo_multimodal_mcp/server.py
```

## Tools

### `understand_image`

Analyze images using Xiaomi MiMo multimodal model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Image understanding task description |
| `image_url` | string | No | Single image URL or data:image base64 |
| `image_path` | string | No | Single local image file path |
| `image_urls` | list[string] | No | Multiple image URLs |
| `image_paths` | list[string] | No | Multiple local image file paths |
| `system_prompt` | string | No | Custom system prompt |
| `max_tokens` | integer | No | Max output length (default: 32768) |

**Supported formats**: JPEG, PNG, GIF, WebP
**Size limit**: 10MB

### `understand_audio`

Analyze audio using Xiaomi MiMo multimodal model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Audio understanding task description |
| `audio_url` | string | No | Single audio URL |
| `audio_path` | string | No | Single local audio file path |
| `audio_urls` | list[string] | No | Multiple audio URLs |
| `audio_paths` | list[string] | No | Multiple local audio file paths |
| `system_prompt` | string | No | Custom system prompt |
| `max_tokens` | integer | No | Max output length (default: 32768) |

**Supported formats**: MP3, WAV, FLAC, M4A, OGG
**Size limit**: URL 100MB, Base64 50MB

### `understand_video`

Analyze video using Xiaomi MiMo multimodal model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Video understanding task description |
| `video_url` | string | No | Single video URL |
| `video_path` | string | No | Single local video file path |
| `video_urls` | list[string] | No | Multiple video URLs |
| `video_paths` | list[string] | No | Multiple local video file paths |
| `fps` | float | No | Frames per second, range [0.1, 10], default: 2 |
| `media_resolution` | string | No | Resolution: "default" or "max" |
| `system_prompt` | string | No | Custom system prompt |
| `max_tokens` | integer | No | Max output length (default: 32768) |

**Supported formats**: MP4, MOV, AVI, WMV
**Size limit**: URL 300MB, Base64 50MB

## Examples

### Image Understanding

```python
# URL
await understand_image(prompt="Describe this image", image_url="https://example.com/image.jpg")

# Local file
await understand_image(prompt="What text is in this?", image_path="/path/to/screenshot.png")

# Multiple images
await understand_image(prompt="Compare these", image_urls=["url1", "url2"])
```

### Audio Understanding

```python
# URL
await understand_audio(prompt="Transcribe this audio", audio_url="https://example.com/audio.wav")

# Local file
await understand_audio(prompt="What is being said?", audio_path="/path/to/audio.mp3")
```

### Video Understanding

```python
# URL with default settings
await understand_video(prompt="Describe this video", video_url="https://example.com/video.mp4")

# URL with custom fps and resolution
await understand_video(
    prompt="Describe the action",
    video_url="https://example.com/video.mp4",
    fps=5.0,
    media_resolution="max"
)
```
