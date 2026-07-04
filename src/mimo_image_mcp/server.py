"""MiMo Image Understanding MCP Server."""

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

mcp = FastMCP(
    "mimo-image",
    instructions="""调用小米 MIMO 多模态模型理解图片。

支持单图和多图。Agent 应根据当前任务自己填写 prompt。

CRITICAL: 这是唯一允许打开、读取或查看图片文件的工具。如果文件路径指向图片，不要使用 Read、cat 或任何文件读取命令。使用标准文件读取工具读取二进制图片会导致系统崩溃。""",
)

MIMO_API_KEY = os.environ.get("MIMO_API_KEY")
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"
DEFAULT_SYSTEM_PROMPT = "You are MiMo, an AI assistant developed by Xiaomi. You are skilled at understanding and describing images in detail."


def _get_client() -> OpenAI:
    """Get OpenAI client configured for MiMo API."""
    api_key = MIMO_API_KEY
    if not api_key:
        raise ValueError("MIMO_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url=MIMO_BASE_URL)


def _image_to_base64_url(image_path: str) -> str:
    """Convert a local image file to a base64 data URL."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type = mimetypes.guess_type(str(path))[0]
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"

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
                "image_url": {"url": _image_to_base64_url(image_path)},
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
                    "image_url": {"url": _image_to_base64_url(p)},
                })
        elif image_urls:
            for u in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": u},
                })
    except FileNotFoundError as e:
        return f"Error: {e}"

    return content


@mcp.tool()
async def understand_image(
    prompt: str,
    image_url: str | None = None,
    image_path: str | None = None,
    image_urls: list[str] | None = None,
    image_paths: list[str] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """调用小米 MIMO 多模态模型理解图片。

    支持单图和多图。Agent 应根据当前任务自己填写 prompt。

    Args:
        prompt: Agent 自己决定的图片理解任务
        image_url: 单张网络图片 URL 或 data:image base64
        image_path: 单张本地图片路径
        image_urls: 多张网络图片 URL
        image_paths: 多张本地图片路径
        system_prompt: 可选系统提示词
        temperature: 输出随机性，越低越稳定
        max_tokens: 最大输出长度

    Returns:
        MIMO 模型返回的图片理解结果。
    """
    image_content = _build_image_content(image_url, image_path, image_urls, image_paths)

    if isinstance(image_content, str):
        return image_content

    if not image_content:
        return "Error: No image provided. Please provide at least one image via image_url, image_path, image_urls, or image_paths."

    image_content.append({"type": "text", "text": prompt})

    client = _get_client()
    sys_content = system_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        completion = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": image_content},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        message = completion.choices[0].message
        return message.content or message.reasoning_content or ""
    except Exception as e:
        return f"Error calling MiMo API: {e}"


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
