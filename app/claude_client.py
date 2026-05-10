from anthropic import Anthropic
from typing import List, Dict, Any
from dotenv import load_dotenv
from app.llm_client import LLMClient, LLMResponse

import os

load_dotenv()


class ClaudeClient(LLMClient):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def send_messages_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse:
        try:
            return self.client.messages.create(
                max_tokens=1028,
                model="claude-opus-4-20250514",
                system=self.system_prompt,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            raise Exception(f"Failed to create message: {str(e)}")
