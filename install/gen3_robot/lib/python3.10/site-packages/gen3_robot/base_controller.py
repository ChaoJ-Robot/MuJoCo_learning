from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseController(ABC):
    def __init__(self, name: str = "BaseController"):
        self.name = name
        print(f"BaseController '{self.name}' initialized.")

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_action(self, action, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def set_up(self, **kwargs: Any) -> None:
        pass

    def get_eef_state(self):
        raise NotImplementedError

    def set_eef_action(self, *args, **kwargs):
        raise NotImplementedError

    def shut_down(self) -> None:
        print(f"BaseController '{self.name}' shutting down.")