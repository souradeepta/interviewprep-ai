"""Bounded conversation state with checkpoint/restore."""

from dataclasses import dataclass, field
import copy


@dataclass
class ConversationState:
    max_messages: int = 8
    messages: list[dict] = field(default_factory=list)
    task_state: dict = field(default_factory=dict)

    def add(self, role, content):
        if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("invalid message")
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages:]

    def checkpoint(self):
        return {"messages": copy.deepcopy(self.messages), "task_state": copy.deepcopy(self.task_state)}

    @classmethod
    def restore(cls, checkpoint, *, max_messages=8):
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("messages"), list):
            raise ValueError("invalid checkpoint")
        state = cls(max_messages=max_messages)
        state.messages = copy.deepcopy(checkpoint["messages"])[-max_messages:]
        state.task_state = copy.deepcopy(checkpoint.get("task_state", {}))
        return state
