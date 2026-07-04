# MiMo Image Understanding MCP Server

MCP server for Xiaomi MiMo image understanding API.

## Features

- Support single and multiple images
- Support image URL and local file paths
- Support Base64 encoded images
- Customizable system prompt, temperature, and max tokens

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
uv run mcp dev src/mimo_image_mcp/server.py
```

### Install to Claude Desktop

```bash
uv run mcp install src/mimo_image_mcp/server.py
```

### Direct execution

```bash
uv run python src/mimo_image_mcp/server.py
```

## Tool

### `understand_image`

Analyze images using Xiaomi MiMo multimodal model.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Image understanding task description |
| `image_url` | string | No | Single image URL or data:image base64 |
| `image_path` | string | No | Single local image file path |
| `image_urls` | list[string] | No | Multiple image URLs |
| `image_paths` | list[string] | No | Multiple local image file paths |
| `system_prompt` | string | No | Custom system prompt |
| `temperature` | float | No | Output randomness (default: 0.7) |
| `max_tokens` | integer | No | Max output length (default: 1024) |

**Example usage in Claude:**

- "Describe this image: https://example.com/image.jpg"
- "What text is in this screenshot? /path/to/screenshot.png"
- "Compare these two images: [url1] [url2]"
