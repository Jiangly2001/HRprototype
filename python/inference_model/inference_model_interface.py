from abc import ABC, abstractmethod
from typing import List, Any

import numpy as np
import torch


class InferenceModelInterface(ABC):
    """
    This class wrap high level applications to interact with Elf.
    For each high level application to run with Elf, it needs to provide the below general interfaces:
    """

    def __init__(self):
        self.app = self.create_model()
        self.predictions = None
        self.attention_weights = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def run(self, img: np.ndarray) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def extract_rps(inference_result: Any) -> np.ndarray:
        pass

    @abstractmethod
    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        pass

    @abstractmethod
    def merge(self, inference_results: List[Any], offsets: List[int], **kwargs) -> Any:
        pass
