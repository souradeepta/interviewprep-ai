"""Bounded conversation state with checkpoint/restore."""

from dataclasses import dataclass, field
import copy
import math
import time


@dataclass
class ConversationState:
    max_messages: int = 8
    messages: list[dict] = field(default_factory=list)
    task_state: dict = field(default_factory=dict)
    completed_action_ids: set[str] = field(default_factory=set)
    clock: object = field(default=time.time, repr=False, compare=False)
    schema_version: int = 1

    def __post_init__(self):
        self._validate_limit(self.max_messages)
        if not isinstance(self.task_state, dict):
            raise ValueError("task_state must be a dictionary")
        if not isinstance(self.completed_action_ids, set) or not all(
                isinstance(action, str) and action for action in self.completed_action_ids):
            raise ValueError("completed_action_ids must be a set of non-empty strings")

    @staticmethod
    def _validate_limit(value):
        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError("max_messages must be a positive integer")

    @staticmethod
    def _validate_messages(messages):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in {"user", "assistant", "tool"}
                    or not isinstance(message.get("content"), str)):
                raise ValueError("invalid checkpoint message")

    def add(self, role, content):
        if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("invalid message")
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages:]

    def mark_completed(self, action_id):
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("action_id must be a non-empty string")
        self.completed_action_ids.add(action_id)

    def checkpoint(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("clock must return a finite number")
        return {
            "schema_version": self.schema_version,
            "checkpoint_time": now,
            "messages": copy.deepcopy(self.messages),
            "task_state": copy.deepcopy(self.task_state),
            "completed_action_ids": sorted(self.completed_action_ids),
        }

    @classmethod
    def restore(cls, checkpoint, *, max_messages=8, clock=time.time, max_checkpoint_age=None):
        cls._validate_limit(max_messages)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
            raise ValueError("invalid checkpoint")
        cls._validate_messages(checkpoint.get("messages"))
        if not isinstance(checkpoint.get("task_state"), dict):
            raise ValueError("task_state must be a dictionary")
        actions = checkpoint.get("completed_action_ids")
        if not isinstance(actions, list) or not all(isinstance(action, str) and action for action in actions):
            raise ValueError("completed_action_ids must be a list of non-empty strings")
        checkpoint_time = checkpoint.get("checkpoint_time")
        now = clock()
        if (isinstance(checkpoint_time, bool) or not isinstance(checkpoint_time, (int, float))
                or not math.isfinite(checkpoint_time) or isinstance(now, bool)
                or not isinstance(now, (int, float)) or not math.isfinite(now)):
            raise ValueError("invalid checkpoint time")
        if max_checkpoint_age is not None:
            if (isinstance(max_checkpoint_age, bool) or not isinstance(max_checkpoint_age, (int, float))
                    or not math.isfinite(max_checkpoint_age) or max_checkpoint_age < 0):
                raise ValueError("max_checkpoint_age must be non-negative and finite")
            if now - checkpoint_time > max_checkpoint_age:
                raise ValueError("checkpoint is stale")
        state = cls(max_messages=max_messages, clock=clock,
                    completed_action_ids=set(actions))
        state.messages = copy.deepcopy(checkpoint["messages"])[-max_messages:]
        state.task_state = copy.deepcopy(checkpoint["task_state"])
        return state
