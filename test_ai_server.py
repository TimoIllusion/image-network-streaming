import pytest
from httpx import AsyncClient
from image_network_streaming.backend.fastapi.ai_server import (
    app,
)  # Import the FastAPI instance from your app module
from image_network_streaming.logging import logger


@pytest.mark.asyncio
async def test_end_to_end():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        with open("resources/example_dall_e.png", "rb") as img:
            response = await ac.post("/detect/", files={"file": img})

    assert response.status_code == 200
    logger.info(f"Response: {response.content}")

    data = response.json()

    detections_single = data["batched_detections"][0]

    assert len(detections_single) > 3, "No detections found in response"
