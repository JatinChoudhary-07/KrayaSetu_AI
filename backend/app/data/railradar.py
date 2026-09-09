import httpx
import asyncio
from typing import Dict, Any

class RailRadarClient:
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(0.5)  # 500ms bounded timeout

    async def get_live_telemetry(self, block_section_id: str) -> Dict[str, Any]:
        """Fetch live telemetry with a strict 500ms bound. Return fallback on failure."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.base_url}/telemetry/{block_section_id}")
                resp.raise_for_status()
                return resp.json()
        except (httpx.TimeoutException, httpx.RequestError):
            # Bounded enrichment fallback
            return {
                "status": "FALLBACK",
                "occupancy": "unknown",
                "last_updated": None
            }
