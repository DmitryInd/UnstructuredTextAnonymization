from abc import ABC, abstractmethod
from typing import List


class Anonymization(ABC):
    """
    Класс для замены уже размеченных сущностей с личными данными синтаксическими данными
    """

    def __init__(self, other_label: str = 'O'):
        self.other_label = other_label  # Метка, которую не нужно заменять

    def __call__(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                 source_text_list: List[List[str]]) -> List[List[str]]:
        """
        :param general_category_list: общие категории сущностей в формате [список категорий слов в документе, ...]
        :param specific_category_list: исходные категории сущностей в формате [список категорий слов в документе, ...]
        :param source_text_list: исходный текст в формате [список слов в документе, ...]
        :return: обезличенный текст в формате [список слов в документе, ...]
        """
        predictions = self._get_substitutions(general_category_list, specific_category_list, source_text_list)
        new_texts = []
        for source_text, categories, answers in zip(source_text_list, general_category_list, predictions):
            new_texts.append([])
            i = 0
            for text, category in zip(source_text, categories):
                if category != self.other_label:
                    if i < len(answers):
                        new_texts[-1].append(answers[i])
                        i += 1
                    else:
                        new_texts[-1].append("")
                else:
                    new_texts[-1].append(text)

        return new_texts

    @abstractmethod
    def _get_substitutions(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                           source_text_list: List[List[str]]) -> List[List[str]]:
        """
        :param general_category_list: общие категории сущностей в формате [список категорий слов в документе, ...]
        :param specific_category_list: исходные категории сущностей в формате [список категорий слов в документе, ...]
        :param source_text_list: исходный текст в формате [список слов в документе, ...]
        :return: список замен для сущностей с личной информацией в хронологическом порядке для каждого документа
        """
        pass
