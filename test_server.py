"""Test script for MiMo Multimodal MCP Server."""

import asyncio
import os


async def test_server():
    """Test the server can be imported and tools are registered."""
    from mimo_multimodal_mcp.server import mcp, understand_image, understand_audio, understand_video

    print("✓ Server imported successfully")
    print(f"✓ Server name: {mcp.name}")

    # Test image tool
    print("\nTesting understand_image tool...")
    result = await understand_image(prompt="describe this image")
    assert "Error" in result, "Should return error when no image provided"
    print(f"✓ No image error: {result[:50]}...")

    result = await understand_image(prompt="describe", image_path="/nonexistent/image.png")
    assert "Error" in result or "not found" in result.lower()
    print(f"✓ Invalid path handled: {result[:50]}...")

    # Test audio tool
    print("\nTesting understand_audio tool...")
    result = await understand_audio(prompt="transcribe this audio")
    assert "Error" in result, "Should return error when no audio provided"
    print(f"✓ No audio error: {result[:50]}...")

    result = await understand_audio(prompt="transcribe", audio_path="/nonexistent/audio.wav")
    assert "Error" in result or "not found" in result.lower()
    print(f"✓ Invalid path handled: {result[:50]}...")

    # Test video tool
    print("\nTesting understand_video tool...")
    result = await understand_video(prompt="describe this video")
    assert "Error" in result, "Should return error when no video provided"
    print(f"✓ No video error: {result[:50]}...")

    result = await understand_video(prompt="describe", video_path="/nonexistent/video.mp4")
    assert "Error" in result or "not found" in result.lower()
    print(f"✓ Invalid path handled: {result[:50]}...")

    print("\n✓ All basic tests passed!")
    print("\nTo test with actual API, set MIMO_API_KEY and run:")
    print("  uv run python test_server.py --live")


async def test_live():
    """Test with actual API call (requires MIMO_API_KEY)."""
    from mimo_multimodal_mcp.server import understand_image, understand_audio, understand_video

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key or api_key == "test":
        print("✗ Set MIMO_API_KEY to a valid key for live testing")
        return

    print("Testing image understanding with live API...")
    result = await understand_image(
        prompt="请描述这张图片的内容",
        image_url="https://example-files.cnbj1.mi-fds.com/example-files/image/image_example.png",
    )
    print(f"✓ Image API Response:\n{result}\n")

    print("Testing audio understanding with live API...")
    result = await understand_audio(
        prompt="请转录这段音频的内容",
        audio_url="https://example-files.cnbj1.mi-fds.com/example-files/audio/audio_example.wav",
    )
    print(f"✓ Audio API Response:\n{result}\n")

    print("Testing video understanding with live API...")
    result = await understand_video(
        prompt="请描述这段视频的内容",
        video_url="https://example-files.cnbj1.mi-fds.com/example-files/video/video_example.mp4",
    )
    print(f"✓ Video API Response:\n{result}")


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        asyncio.run(test_live())
    else:
        asyncio.run(test_server())
