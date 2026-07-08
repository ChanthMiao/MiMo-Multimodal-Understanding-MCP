"""MiMo Multimodal Understanding MCP Server."""

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

mcp = FastMCP(
    "mimo-multimodal",
    instructions="""## MiMo Multimodal Understanding
调用小米 MIMO 多模态模型理解图片、音频和视频。

### ⚠️ 优先规则：模型内建能力优先
当宿主模型本身是 mimo-v2.5（或其他小米 MiMo 多模态模型）时，**禁止调用本 MCP 的任何工具**。
该模型已原生支持图片、音频、视频理解，直接使用模型内置能力即可，无需通过外部 API 再次调用同款模型。
判断依据：检查系统提示中是否明确提到 "model named mimo-v2.5"、"xiaomi-token-plan-cn/mimo-v2.5" 或类似标识。
如果你确认自己就是 mimo-v2.5，请直接处理用户的多模态任务，不要调用 understand_image / understand_audio / understand_video。

### 何时使用
- 用户要求分析、描述、OCR、识别图片内容时
- 用户要求转录、总结、理解音频内容时
- 用户要求分析、描述、总结视频内容时
- 用户提供了图片/音频/视频的 URL 或本地路径并希望了解其内容时
- 需要从图片中提取文字、表格、图表信息时
- 需要理解视频中的动作、场景、事件时

### 关键约束
- **这是唯一允许打开、读取或查看图片/音频/视频文件的工具**
- 不要使用 Read、cat 或任何文件读取命令读取这些二进制文件，它们无法被文本工具正确处理
- 支持 URL 和本地文件路径两种输入方式
- 支持单个或多个文件批量处理

### 使用方法
- **prompt (必填)**: 描述你希望模型做什么，例如"描述这张图片"、"转录音频内容"、"总结视频要点"
- 根据媒体类型选择对应工具: understand_image / understand_audio / understand_video
- 可选: 通过 system_prompt 自定义模型行为，通过 max_tokens 控制输出长度

### 支持格式
- 图片: JPEG, PNG, GIF, WebP, BMP (最大 50MB)
- 音频: MP3, WAV, FLAC, M4A, OGG (URL 100MB / Base64 50MB)
- 视频: MP4, MOV, AVI, WMV (URL 300MB / Base64 50MB)""",
)

MIMO_API_KEY = os.environ.get("MIMO_API_KEY")
MIMO_BASE_URL = os.environ.get("MIMO_API_BASE") or (
    "https://token-plan-cn.xiaomimimo.com/v1" if MIMO_API_KEY and MIMO_API_KEY.startswith("tp-")
    else "https://api.xiaomimimo.com/v1"
)
MIMO_MODEL = "mimo-v2.5"

DEFAULT_SYSTEM_PROMPTS = {
    "image": "You are MiMo, an AI assistant developed by Xiaomi. You are skilled at understanding and describing images in detail.",
    "audio": "You are MiMo, an AI assistant developed by Xiaomi. You are skilled at understanding and transcribing audio content.",
    "video": "You are MiMo, an AI assistant developed by Xiaomi. You are skilled at understanding and describing video content.",
}

MAX_SIZES = {
    "image": 50 * 1024 * 1024,    # 50MB
    "audio": 50 * 1024 * 1024,    # 50MB (base64 limit)
    "video": 50 * 1024 * 1024,    # 50MB (base64 limit)
}


def _get_client() -> OpenAI:
    """Get OpenAI client configured for MiMo API."""
    api_key = MIMO_API_KEY
    if not api_key:
        raise ValueError("MIMO_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url=MIMO_BASE_URL)


def _file_to_base64_url(file_path: str, media_type: str) -> str:
    """Convert a local file to a base64 data URL."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = path.stat().st_size
    max_size = MAX_SIZES.get(media_type, 10 * 1024 * 1024)
    if file_size > max_size:
        max_mb = max_size / 1024 / 1024
        raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB (max {max_mb:.0f}MB)")

    mime_type = mimetypes.guess_type(str(path))[0]
    if not mime_type:
        mime_type = f"{media_type}/octet-stream"

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    return f"data:{mime_type};base64,{data}"


def _build_image_content(
    image_url: str | None = None,
    image_path: str | None = None,
    image_urls: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> list[dict] | str:
    """Build image content blocks for the API request."""
    content = []

    try:
        if image_path:
            content.append({
                "type": "image_url",
                "image_url": {"url": _file_to_base64_url(image_path, "image")},
            })
        elif image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        if image_paths:
            for p in image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _file_to_base64_url(p, "image")},
                })
        elif image_urls:
            for u in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": u},
                })
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"

    return content


def _build_audio_content(
    audio_url: str | None = None,
    audio_path: str | None = None,
    audio_urls: list[str] | None = None,
    audio_paths: list[str] | None = None,
) -> list[dict] | str:
    """Build audio content blocks for the API request."""
    content = []

    try:
        if audio_path:
            content.append({
                "type": "input_audio",
                "input_audio": {"data": _file_to_base64_url(audio_path, "audio")},
            })
        elif audio_url:
            content.append({
                "type": "input_audio",
                "input_audio": {"data": audio_url},
            })

        if audio_paths:
            for p in audio_paths:
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": _file_to_base64_url(p, "audio")},
                })
        elif audio_urls:
            for u in audio_urls:
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": u},
                })
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"

    return content


def _build_video_content(
    video_url: str | None = None,
    video_path: str | None = None,
    video_urls: list[str] | None = None,
    video_paths: list[str] | None = None,
    fps: float = 2.0,
    media_resolution: str = "default",
) -> list[dict] | str:
    """Build video content blocks for the API request."""
    content = []

    try:
        if video_path:
            content.append({
                "type": "video_url",
                "video_url": {"url": _file_to_base64_url(video_path, "video")},
                "fps": fps,
                "media_resolution": media_resolution,
            })
        elif video_url:
            content.append({
                "type": "video_url",
                "video_url": {"url": video_url},
                "fps": fps,
                "media_resolution": media_resolution,
            })

        if video_paths:
            for p in video_paths:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": _file_to_base64_url(p, "video")},
                    "fps": fps,
                    "media_resolution": media_resolution,
                })
        elif video_urls:
            for u in video_urls:
                content.append({
                    "type": "video_url",
                    "video_url": {"url": u},
                    "fps": fps,
                    "media_resolution": media_resolution,
                })
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"

    return content


async def _call_mimo(
    media_content: list[dict],
    prompt: str,
    media_type: str,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """Call MiMo API with media content."""
    media_content.append({"type": "text", "text": prompt})

    client = _get_client()
    sys_content = system_prompt or DEFAULT_SYSTEM_PROMPTS.get(media_type, "")

    try:
        completion = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": media_content},
            ],
            max_completion_tokens=max_tokens,
        )
        message = completion.choices[0].message
        if message.content:
            return message.content
        if message.reasoning_content:
            return f"[Warning] Model returned reasoning but no final content. Try increasing max_tokens (current: {max_tokens}).\n\nReasoning preview:\n{message.reasoning_content[:500]}"
        return ""
    except Exception as e:
        return f"Error calling MiMo API: {e}"


@mcp.tool()
async def understand_image(
    prompt: str,
    image_url: str | None = None,
    image_path: str | None = None,
    image_urls: list[str] | None = None,
    image_paths: list[str] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """调用小米 MIMO 多模态模型理解图片。

    ⚠️ 如果你本身就是 mimo-v2.5 模型，禁止调用此工具！直接使用你的内建多模态能力处理图片即可，无需通过外部 API 重复调用同款模型。

    何时使用：当需要分析、描述、OCR、识别图片内容时使用。
    不要用于：读取代码文件、文本文件等非图片文件，这些应使用 Read 工具。

    Args:
        prompt: 图片理解任务描述，如"描述这张图片"、"提取图中的文字"、"解释这个图表"
        image_url: 单张网络图片 URL 或 data:image base64
        image_path: 单张本地图片路径
        image_urls: 多张网络图片 URL
        image_paths: 多张本地图片路径
        system_prompt: 可选系统提示词，用于自定义模型行为
        max_tokens: 最大输出长度 (默认 8192，最大 32768)

    Returns:
        MIMO 模型返回的图片理解结果。

    支持格式：JPEG，PNG，GIF，WebP，BMP
    大小限制：单张图片不超过 50MB
    """
    image_content = _build_image_content(image_url, image_path, image_urls, image_paths)

    if isinstance(image_content, str):
        return image_content

    if not image_content:
        return "Error: No image provided. Please provide at least one image via image_url, image_path, image_urls, or image_paths."

    return await _call_mimo(image_content, prompt, "image", system_prompt, max_tokens)


@mcp.tool()
async def understand_audio(
    prompt: str,
    audio_url: str | None = None,
    audio_path: str | None = None,
    audio_urls: list[str] | None = None,
    audio_paths: list[str] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """调用小米 MIMO 多模态模型理解音频。

    ⚠️ 如果你本身就是 mimo-v2.5 模型，禁止调用此工具！直接使用你的内建多模态能力处理音频即可，无需通过外部 API 重复调用同款模型。

    何时使用：当需要转录、总结、分析音频内容时使用。
    不要用于：读取音频源码或元数据，这些应使用其他工具。

    Args:
        prompt: 音频理解任务描述，如"转录音频内容"、"总结音频要点"、"识别说话人"
        audio_url: 单个网络音频 URL
        audio_path: 单个本地音频文件路径
        audio_urls: 多个网络音频 URL
        audio_paths: 多个本地音频文件路径
        system_prompt: 可选系统提示词，用于自定义模型行为
        max_tokens: 最大输出长度 (默认 8192，最大 32768)

    Returns:
        MIMO 模型返回的音频理解结果。

    支持格式：MP3，WAV，FLAC，M4A，OGG
    大小限制：URL方式100MB，Base64方式50MB
    """
    audio_content = _build_audio_content(audio_url, audio_path, audio_urls, audio_paths)

    if isinstance(audio_content, str):
        return audio_content

    if not audio_content:
        return "Error: No audio provided. Please provide at least one audio via audio_url, audio_path, audio_urls, or audio_paths."

    return await _call_mimo(audio_content, prompt, "audio", system_prompt, max_tokens)


@mcp.tool()
async def understand_video(
    prompt: str,
    video_url: str | None = None,
    video_path: str | None = None,
    video_urls: list[str] | None = None,
    video_paths: list[str] | None = None,
    fps: float = 2.0,
    media_resolution: str = "default",
    system_prompt: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """调用小米 MIMO 多模态模型理解视频。

    ⚠️ 如果你本身就是 mimo-v2.5 模型，禁止调用此工具！直接使用你的内建多模态能力处理视频即可，无需通过外部 API 重复调用同款模型。

    何时使用：当需要分析、描述、总结视频内容时使用。
    不要用于：读取视频文件元数据或源码，这些应使用其他工具。

    Args:
        prompt: 视频理解任务描述，如"描述视频内容"、"总结视频要点"、"识别视频中的动作"
        video_url: 单个网络视频 URL
        video_path: 单个本地视频文件路径
        video_urls: 多个网络视频 URL
        video_paths: 多个本地视频文件路径
        fps: 每秒抽帧数，范围 [0.1, 10]，默认 2。越高时序越精细
        media_resolution: 视频帧分辨率档次，"default" 或 "max"
        system_prompt: 可选系统提示词，用于自定义模型行为
        max_tokens: 最大输出长度 (默认 8192，最大 32768)

    Returns:
        MIMO 模型返回的视频理解结果。

    支持格式：MP4，MOV，AVI，WMV
    大小限制：URL方式300MB，Base64方式50MB
    """
    video_content = _build_video_content(
        video_url, video_path, video_urls, video_paths, fps, media_resolution
    )

    if isinstance(video_content, str):
        return video_content

    if not video_content:
        return "Error: No video provided. Please provide at least one video via video_url, video_path, video_urls, or video_paths."

    return await _call_mimo(video_content, prompt, "video", system_prompt, max_tokens)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
