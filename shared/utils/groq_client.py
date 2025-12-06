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
    
    async def generate_extraction(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Generate structured extraction using Groq API (replaces Ollama extraction).
        
        Args:
            text: Text to extract from
            system_prompt: System instruction
            user_prompt: User prompt with context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response with triples
        """
        logger.debug(f"Extraction request: {len(text)} chars")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                stream=False,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            parsed = json.loads(answer)
            
            logger.debug(f"Extracted {len(parsed.get('triples', []))} triples")
            return parsed
            
        except Exception as e:
            logger.error(f"Extraction generation failed: {e}", exc_info=True)
            raise
    
    async def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Generate reasoning response using Groq API (replaces Ollama reasoning).
        
        Args:
            system_prompt: System instruction
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed JSON response
        """
        logger.debug("Reasoning request")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                stream=False,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            parsed = json.loads(answer)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}", exc_info=True)
            raise
    
    async def generate_cypher(
        self,
        question: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Generate Cypher query from natural language using Groq API (NL2Cypher).
        
        This is a CORE feature for the conference paper:
        "Querying property graphs with natural language interfaces powered by LLMs"
        
        Args:
            question: Natural language question
            system_prompt: NL2CYPHER_SYSTEM_PROMPT with schema
            user_prompt: Formatted NL2CYPHER_USER_TEMPLATE
            temperature: Sampling temperature (low for precise queries)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with:
                - cypher: Generated Cypher query string
                - parameters: Query parameters
                - explanation: What the query does
        """
        logger.debug(f"NL2Cypher request: {question[:100]}")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                stream=False,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            parsed = json.loads(answer)
            
            # Validate required fields
            if "cypher" not in parsed:
                raise ValueError("Generated response missing 'cypher' field")
            
            logger.info(
                f"NL2Cypher generated query: {parsed.get('explanation', 'N/A')}"
            )
            logger.debug(f"Cypher: {parsed['cypher']}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"NL2Cypher generation failed: {e}", exc_info=True)
            raise


@lru_cache(maxsize=1)
def get_groq_client() -> GroqClient:
    """Get cached Groq client instance."""
    return GroqClient()
