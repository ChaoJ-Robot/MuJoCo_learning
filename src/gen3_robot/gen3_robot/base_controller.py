import sys
import os

current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_script_dir)
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union
import sys
import os

current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_script_dir)

#  抽象的基类。继承自ABC，必须重构BaseController类中的抽象方法。
class BaseController(ABC):
    """
    Base class for all controllers.
    Defines abstract methods that must be implemented by concrete subclasses.
    """

    def __init__(self, name: str = "BaseController"):
        """
        Constructor for the BaseController.
        Subclasses should call super().__init__() if they override __init__.
        """
        self.name = name
        print(f"BaseController '{self.name}' initialized.")

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Abstract method to retrieve the current state of the controlled system.
        This method must be implemented by any concrete subclass.
        Returns:
            Dict[str, Any]: A dictionary representing the current state.
                            Example: {"joint_positions": np.array([...]), "gripper_state": "open"}
        """
        pass  # Abstract methods typically have an empty body

    @abstractmethod
    def set_action(self, action: list, **kwargs: Any) -> None:
        """
        Abstract method to apply an action to the controlled system.
        This method must be implemented by any concrete subclass.
        Args:
            action list: The action to be applied.

        """
        pass

    @abstractmethod
    def set_up(self, **kwargs: Any) -> None:
        """
        Abstract method for setting up the controller.
        This method must be implemented by any concrete subclass.
        It's intended for one-time initialization, such as connecting to hardware,
        loading parameters, etc.
        Args:
            None
        """
        pass

    # 不一定要有，选择重载
    def get_eef_state(self):
        pass

    def set_eef_action(self):
        pass

    def shut_down(self) -> None:
        """
        Optional method for graceful shutdown.
        Subclasses can override this if specific cleanup is needed.
        """
        print(f"BaseController '{self.name}' shutting down.")
