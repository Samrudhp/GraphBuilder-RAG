"""
Ollama client wrapper for LLM operations.

Handles:
- DeepSeek-R1-Distill-Qwen-1.5B for extraction
- DeepSeek-R1-Distill-LLaMA-7B for reasoning/QA
- JSON parsing and validation
- Retry logic
- Token management
"""
import json
import logging
from typing import Any, Optional

import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM models."""
    
    def __init__(self):
        self.settings = get_settings().ollama
        self.client = ollama.Client(host=self.settings.base_url)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def generate_extraction(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate structured extraction using DeepSeek-R1-Distill-Qwen-1.5B.
        
        Args:
            text: Text to extract from
            system_prompt: System instruction
            user_prompt: User prompt with context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response
        """
        logger.debug(f"Extraction request: {len(text)} chars")
        
        try:
            response = self.client.chat(
                model=self.settings.extraction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            
            # Extract response text
            response_text = response["message"]["content"]
            
            # Parse JSON (strict)
            try:
                result = self._parse_json_response(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}\nResponse: {response_text}")
                # Attempt to extract JSON from markdown code blocks
                result = self._extract_json_from_markdown(response_text)
                if result:
                    return result
                raise ValueError(f"Failed to parse JSON: {e}")
                
        except Exception as e:
            logger.error(f"Extraction generation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate reasoning response using DeepSeek-R1-Distill-LLaMA-7B.
        
        Args:
            system_prompt: System instruction
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response
        """
        logger.debug(f"Reasoning request")
        
        try:
            response = self.client.chat(
                model=self.settings.reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            
            response_text = response["message"]["content"]
            
            # Parse JSON
            try:
                result = self._parse_json_response(response_text)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}\nResponse: {response_text}")
                result = self._extract_json_from_markdown(response_text)
                if result:
                    return result
                raise ValueError(f"Failed to parse JSON: {e}")
                
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            raise
    
    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from response text."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        return json.loads(text)
    
    def _extract_json_from_markdown(self, text: str) -> Optional[dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        import re
        
        # Find JSON in code blocks
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{.*?\})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            models = self.client.list()
            available_models = [m["name"] for m in models["models"]]
            return model_name in available_models
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def ensure_models_available(self):
        """Ensure required models are pulled."""
        required_models = [
            self.settings.extraction_model,
            self.settings.reasoning_model,
        ]
        
        for model in required_models:
            if not self.check_model_availability(model):
                logger.warning(f"Model not found: {model}")
                logger.info(f"Please run: ollama pull {model}")
                raise RuntimeError(
                    f"Required model not available: {model}\n"
                    f"Run: ollama pull {model}"
                )
        
        logger.info("All required models are available")


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
