from abc import ABC, abstractmethod
from typing import List


class Anonymization(ABC):
    """
    Класс для замены уже размеченных сущностей с личными данными синтаксическими данными
    """
    @abstractmethod
    def __call__(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                 source_text_list: List[List[str]]) -> List[List[str]]:
        """
        :param general_category_list: общие категории сущностей в формате [список категорий слов в документе, ...]
        :param specific_category_list: исходные категории сущностей в формате [список категорий слов в документе, ...]
        :param source_text_list: исходный текст в формате [список слов в документе, ...]
        :return: обезличенный текст в формате [список слов в документе, ...]
        """
        pass
