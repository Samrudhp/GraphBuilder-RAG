"""
Groq Cloud API Client

Fast inference using Groq's cloud API for query answering.
"""
import logging
from functools import lru_cache
from typing import Optional

from groq import AsyncGroq

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for Groq Cloud API."""
    
    def __init__(self):
        self.settings = get_settings().groq
        
        if not self.settings.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = AsyncGroq(api_key=self.settings.api_key)
        self.model = self.settings.model
        
        logger.info(f"Groq client initialized with model: {self.model}")
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate response using Groq API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query/prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Max tokens to generate
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Use settings defaults if not provided
            temp = temperature if temperature is not None else self.settings.temperature
            max_tok = max_tokens if max_tokens is not None else self.settings.max_tokens
            
            # Call Groq API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                top_p=1,
                stream=False,
            )
            
            # Extract response
            answer = response.choices[0].message.content.strip()
            
            # Get usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            logger.debug(f"Groq response generated: {usage['total_tokens']} tokens")
            
            return {
                "response": answer,
                "usage": usage,
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason,
            }
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}", exc_info=True)
            raise
    
    async def generate_with_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate JSON response using Groq API.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query/prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Dict with parsed JSON response
        """
        try:
            temp = temperature if temperature is not None else self.settings.temperature
            max_tok = max_tokens if max_tokens is not None else self.settings.max_tokens
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                response_format={"type": "json_object"},
                stream=False,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            parsed = json.loads(answer)
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            logger.debug(f"Groq JSON response generated: {usage['total_tokens']} tokens")
            
            return {
                "response": parsed,
                "usage": usage,
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason,
            }
            
        except Exception as e:
            logger.error(f"Groq JSON API call failed: {e}", exc_info=True)
            raise


@lru_cache(maxsize=1)
def get_groq_client() -> GroqClient:
    """Get cached Groq client instance."""
    return GroqClient()
