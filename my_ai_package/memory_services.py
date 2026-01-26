"""
Memory Services - URL building, health checking, and LLM completion utilities.

Provides abstracted access to dual LLM setup with fallback logic:
- Port 8000: Mistral-7B (primary LLM)
- Port 8001: Stable-Code-3B (small LLM)
- Port 8002: Reranker
- Port 8008: Embeddings
"""

import time
from typing import Optional

import httpx

from my_ai_package import config
from my_ai_package.memory_logging import log_info, log_error, log_debug, log_warning


class MemoryServices:
    """
    Service layer for memory system operations.

    Handles URL construction, health checking with TTL cache,
    and LLM completion with fallback logic.
    """

    def __init__(self, pod_id: str, api_key: str):
        """
        Initialize memory services.

        Args:
            pod_id: RunPod pod identifier
            api_key: API key for authentication
        """
        self.pod_id = pod_id
        self.api_key = api_key
        self._health_cache: dict[str, tuple[bool, float]] = {}

    # =========================================================================
    # URL Builders
    # =========================================================================

    def primary_llm_url(self) -> str:
        """Get primary LLM URL (Mistral-7B on port 8000)."""
        return f"https://{self.pod_id}-8000.proxy.runpod.net"

    def small_llm_url(self) -> str:
        """Get small LLM URL (Stable-Code-3B on port 8001)."""
        return f"https://{self.pod_id}-8001.proxy.runpod.net"

    def embeddings_url(self) -> str:
        """Get embeddings service URL (port 8008)."""
        return f"https://{self.pod_id}-8008.proxy.runpod.net"

    def reranker_url(self) -> str:
        """Get reranker service URL (port 8002)."""
        return f"https://{self.pod_id}-8002.proxy.runpod.net"

    # =========================================================================
    # Health Checking
    # =========================================================================

    def is_healthy(self, service: str, ttl_sec: int = 30) -> bool:
        """
        Check if a service is healthy with TTL cache.

        Args:
            service: Service name ('primary_llm', 'small_llm', 'embeddings', 'reranker')
            ttl_sec: Cache TTL in seconds (default 30)

        Returns:
            True if service is healthy, False otherwise
        """
        # Check cache
        if service in self._health_cache:
            is_healthy, cached_at = self._health_cache[service]
            if time.time() - cached_at < ttl_sec:
                return is_healthy

        # Get URL for service
        url_map = {
            "primary_llm": self.primary_llm_url,
            "small_llm": self.small_llm_url,
            "embeddings": self.embeddings_url,
            "reranker": self.reranker_url,
        }

        if service not in url_map:
            return False

        base_url = url_map[service]()

        # Check health (use /v1/models for LLMs, /health for others)
        if service in ("primary_llm", "small_llm"):
            endpoint = f"{base_url}/v1/models"
        else:
            endpoint = f"{base_url}/health"

        try:
            response = httpx.get(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            is_healthy = response.status_code == 200
        except Exception as e:
            is_healthy = False
            log_debug(f"Health check failed for {service}: {e}")

        # Update cache
        self._health_cache[service] = (is_healthy, time.time())
        log_debug(f"Health check {service}: {'healthy' if is_healthy else 'unhealthy'}")
        return is_healthy

    # =========================================================================
    # LLM Completion
    # =========================================================================

    def llm_complete(
        self,
        prompt: str,
        prefer_small: Optional[bool] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Call LLM with fallback logic.

        Args:
            prompt: The prompt to send to the LLM
            prefer_small: If True, try small LLM first. Defaults to config.PREFER_SMALL_LLM
            max_tokens: Maximum tokens to generate (default 1000)
            temperature: Sampling temperature (default 0.3)

        Returns:
            Model response text, or None if all attempts fail
        """
        if prefer_small is None:
            prefer_small = config.PREFER_SMALL_LLM

        # Build order based on preference
        if prefer_small:
            order = [
                ("small_llm", self.small_llm_url()),
                ("primary_llm", self.primary_llm_url()),
            ]
        else:
            order = [
                ("primary_llm", self.primary_llm_url()),
                ("small_llm", self.small_llm_url()),
            ]

        # Try each LLM in order
        for service_name, base_url in order:
            if not self.is_healthy(service_name):
                log_debug(f"Skipping {service_name} (unhealthy)")
                continue

            log_debug(f"Trying {service_name} at {base_url}")
            result = self._call_llm(base_url, prompt, max_tokens, temperature)
            if result is not None:
                log_info(f"LLM call success via {service_name}")
                return result
            else:
                log_warning(f"LLM call failed via {service_name}")

        log_error("All LLM endpoints failed")
        return None

    def _call_llm(
        self,
        base_url: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Make a single LLM completion call.

        Args:
            base_url: LLM server base URL
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Response text or None on failure
        """
        start_time = time.time()
        try:
            response = httpx.post(
                f"{base_url}/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=60,
            )
            elapsed = time.time() - start_time

            if response.status_code != 200:
                log_error(f"LLM returned HTTP {response.status_code} ({elapsed:.1f}s)")
                return None

            data = response.json()
            log_debug(f"LLM response received ({elapsed:.1f}s)")
            return data["choices"][0]["text"].strip()

        except Exception as e:
            elapsed = time.time() - start_time
            log_error(f"LLM call exception ({elapsed:.1f}s): {e}")
            return None
