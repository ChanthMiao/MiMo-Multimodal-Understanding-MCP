"""Test script for MiMo Image MCP Server."""

import asyncio
import os

from mcp.server.fastmcp import FastMCP


async def test_server():
    """Test the server can be imported and tools are registered."""
    from mimo_image_mcp.server import mcp, understand_image

    print("✓ Server imported successfully")
    print(f"✓ Server name: {mcp.name}")

    # Test tool function directly
    print("\nTesting understand_image tool...")

    # Test with no image (should return error)
    result = await understand_image(prompt="describe this image")
    assert "Error" in result, "Should return error when no image provided"
    print(f"✓ No image error: {result[:50]}...")

    # Test with invalid local path
    result = await understand_image(
        prompt="describe",
        image_path="/nonexistent/image.png"
    )
    assert "Error" in result or "not found" in result.lower()
    print(f"✓ Invalid path handled: {result[:50]}...")

    print("\n✓ All basic tests passed!")
    print("\nTo test with actual API, set MIMO_API_KEY and run:")
    print("  uv run python test_server.py --live")


async def test_live():
    """Test with actual API call (requires MIMO_API_KEY)."""
    from mimo_image_mcp.server import understand_image

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key or api_key == "test":
        print("✗ Set MIMO_API_KEY to a valid key for live testing")
        return

    print("Testing with live API...")
    result = await understand_image(
        prompt="请描述这张图片的内容",
        image_url="https://example-files.cnbj1.mi-fds.com/example-files/image/image_example.png",
        max_tokens=512
    )
    print(f"✓ API Response:\n{result}")


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        asyncio.run(test_live())
    else:
        asyncio.run(test_server())
