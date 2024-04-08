from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from typing import List, Tuple


class MaskFn(ABC):
    """
    Класс для поиска/генерации масок, скрывающих частей исходного текста
    """
    @property
    @abstractmethod
    def mask_types(self) -> List[Enum]:
        raise NotImplementedError()

    @abstractmethod
    def mask(self, doc) -> Tuple[List[List[Tuple[Enum, int, int]]], Counter]:
        raise NotImplementedError()
