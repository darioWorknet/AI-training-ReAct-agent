from abc import ABC, abstractmethod
from typing import Any, List, Dict, Protocol, runtime_checkable


@runtime_checkable
class ContentBlock(Protocol):
    type: str


@runtime_checkable
class LLMResponse(Protocol):
    stop_reason: str
    content: List[Any]


class LLMClient(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    def send_messages_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMResponse: ...
