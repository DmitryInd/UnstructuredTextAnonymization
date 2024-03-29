from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class MaskFn(ABC):
    """
    Класс для поиска/генерации масок, скрывающих частей исходного текста
    """
    @staticmethod
    @abstractmethod
    def mask_types() -> List[Enum]:
        raise NotImplementedError()

    @abstractmethod
    def mask_type_serialize(self, m_type: Enum) -> str:
        raise NotImplementedError()

    @abstractmethod
    def mask(self, doc):
        raise NotImplementedError()
