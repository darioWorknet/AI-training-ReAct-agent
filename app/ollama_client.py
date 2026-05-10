import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from app.llm_client import LLMClient, LLMResponse

load_dotenv()


class _ContentBlock:
    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Message:
    def __init__(self, stop_reason: str, content: list):
        self.stop_reason = stop_reason
        self.content = content


class OllamaClient(LLMClient):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.12:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")

    def send_messages_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResponse:
        try:
            openai_messages = [{"role": "system", "content": self.system_prompt}]
            openai_messages += self._convert_messages(messages)
            openai_tools = self._convert_tools(tools)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                max_tokens=1028,
            )
            return self._convert_response(response)
        except Exception as e:
            raise Exception(f"Failed to create message: {str(e)}")

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        openai_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list) and content:
                first = content[0]
                if first.get("type") == "tool_use":
                    tool_calls = [
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        }
                        for block in content
                        if block.get("type") == "tool_use"
                    ]
                    openai_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                    )
                elif first.get("type") == "tool_result":
                    for block in content:
                        if block.get("type") == "tool_result":
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": block["content"],
                                }
                            )
        return openai_messages

    def _convert_response(self, response) -> _Message:
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and message.tool_calls:
            content = [
                _ContentBlock(
                    type="tool_use",
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                    id=tc.id,
                )
                for tc in message.tool_calls
            ]
            return _Message(stop_reason="tool_use", content=content)

        text = message.content or ""
        return _Message(
            stop_reason="end_turn",
            content=[_ContentBlock(type="text", text=text)],
        )
