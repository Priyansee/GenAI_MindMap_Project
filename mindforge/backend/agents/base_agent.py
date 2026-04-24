import json
import httpx
import os
import re
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(basedir, '.env'))

import asyncio

class BaseAgent:
    def __init__(self, name: str, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.name = name
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """To be implemented by subclasses"""
        raise NotImplementedError

    async def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        if not self.api_key:
            raise ValueError(f"API key missing for agent: {self.name}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"You are the {self.name} assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=45.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 429:
                        wait_time = (attempt + 1) * 2
                        print(f"XXRATE LIMIT (429) for {self.name}. Retrying in {wait_time}s...XX")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        print(f"ERROR FROM {self.name}: {response.status_code} - {response.text}")
                    
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 430 and attempt < max_retries - 1:
                        continue
                    print(f"HTTP Error from {self.name}: {e.response.text}")
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    raise
        
        raise Exception(f"Failed to get response from {self.name} after {max_retries} attempts due to rate limits.")

    def parse_json(self, raw: str) -> Dict[str, Any]:
        try:
            # More robust regex to find the outermost JSON object
            # It looks for the first '{' and the last '}' in the entire string
            start_index = raw.find('{')
            end_index = raw.rfind('}')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = raw[start_index:end_index + 1]
                return json.loads(json_str)
            
            # Fallback to standard stripping
            clean_raw = raw.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]
            
            return json.loads(clean_raw.strip())
        except Exception as e:
            print(f"Error parsing JSON from {self.name}: {e}")
            # Log a snippet of the raw output to see what went wrong
            print(f"Raw output (first 100 chars): {raw[:100]}...")
            raise
