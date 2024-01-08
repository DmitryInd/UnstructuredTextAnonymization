from abc import ABC, abstractmethod
from typing import List


class MaskFn(ABC):
    @staticmethod
    @abstractmethod
    def mask_types():
        raise NotImplementedError()

    @abstractmethod
    def mask_type_serialize(self, m_type):
        raise NotImplementedError()

    @abstractmethod
    def mask(self, doc):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def mask_types() -> List[str]:
        pass
