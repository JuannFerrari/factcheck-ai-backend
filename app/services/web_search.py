import httpx
import asyncio
from typing import List, Dict, Any
from app.domain.models import Source
from app.core.config import settings


class WebSearchService:
    def __init__(self):
        self.api_key = settings.serper_api_key
        self.base_url = "https://google.serper.dev/search"

    async def search_claim(self, claim: str, num_results: int = 5) -> List[Source]:
        """
        Search for information related to a claim using Serper.dev API
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "q": claim,
                        "num": num_results
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                sources = []
                if "organic" in data:
                    for result in data["organic"][:num_results]:
                        source = Source(
                            title=result.get("title", "No title"),
                            url=result.get("link", ""),
                            snippet=result.get("snippet", "")
                        )
                        sources.append(source)

                return sources

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in web search: {e}")
            return []


# Global instance
web_search_service = WebSearchService()
