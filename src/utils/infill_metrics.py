import random
from typing import List, Tuple

import numpy as np
from tabulate import tabulate
from torchmetrics.functional.text import char_error_rate

from anonymization.base import Anonymization


class Statistics:
    def __init__(self, anonymization: Anonymization,
                 general_category_list: List[List[str]],
                 specific_category_list: List[List[str]],
                 source_text_list: List[List[str]],
                 is_uncased=True):
        """
        :param anonymization: инструмент для генерации синтетических замен
        :param general_category_list: общие категории сущностей в формате [список категорий слов в документе, ...]
        :param specific_category_list: исходные категории сущностей в формате [список категорий слов в документе, ...]
        :param source_text_list: исходный текст в формате [список слов в документе, ...]
        :param is_uncased: учитывать ли в регистр символов при подсчёте ошибок
        """
        self.anonymization = anonymization
        self.general_category_list = general_category_list
        self.specific_category_list = specific_category_list
        self.source_text_list = source_text_list
        self.is_uncased = is_uncased

        self.other_label = self.anonymization.other_label
        self.completed_sentences = self.anonymization(general_category_list,
                                                      specific_category_list,
                                                      source_text_list)
        self.error_rates = self._calculate_cer()
        self.ex_cer = np.array(list(map(np.mean, self.error_rates)))
        self.avg_cer = self.ex_cer.mean()

    def _calculate_cer(self) -> List[List[float]]:
        cer = []
        # Sub - substituted
        for labels, doc, sub_doc in zip(self.general_category_list, self.source_text_list, self.completed_sentences):
            doc_cer = []
            for label, section, sub_section in zip(labels, doc, sub_doc):
                if label != self.other_label:
                    doc_cer.append(char_error_rate(sub_section.lower() if self.is_uncased else sub_section,
                                                   section.lower() if self.is_uncased else section).item())
            cer.append(doc_cer)

        return cer

    def examples_by_indexes(self, indexes) \
            -> Tuple[List[List[float]], List[List[str]], List[List[str]], List[List[str]]]:
        source = [self.source_text_list[i] for i in indexes]
        substituted = [self.completed_sentences[i] for i in indexes]
        labels = [self.specific_category_list[i] for i in indexes]
        cer = [self.error_rates[i] for i in indexes]
        return cer, labels, source, substituted

    def random_examples_indexes(self, n) -> List[int]:
        indexes = random.sample(list(range(len(self.source_text_list))), n)
        return indexes

    def most_close_examples_indexes(self, n) -> List[int]:
        indexes = np.argsort(self.ex_cer)[:n].tolist()
        return indexes

    def most_distant_examples_indexes(self, n) -> List[int]:
        indexes = np.argsort(self.ex_cer)[-n:][::-1].tolist()
        return indexes

    def _print_examples(self, error_rate: List[List[float]],
                        labels: List[List[str]], source_texts: List[List[str]], substituted: List[List[str]],
                        indexes=None):
        for i in range(len(source_texts)):
            cursor = 0
            aligned_error_rate = []
            for label in labels[i]:
                if label != self.other_label and cursor < len(error_rate[i]):
                    aligned_error_rate.append(error_rate[i][cursor])
                    cursor += 1
                else:
                    aligned_error_rate.append("")

            print('_' * 5 + f' Record {i if indexes is None else indexes[i]} ' + '_' * 5)
            print(tabulate(
                [
                    ['Labels:'] + labels[i],
                    ['Source text:'] + source_texts[i],
                    ['Substituted text:'] + substituted[i],
                    ['CER'] + aligned_error_rate
                ],
                tablefmt='orgtbl'))

    def print_examples_by_indexes(self, indexes):
        cer, labels, source, substituted = self.examples_by_indexes(indexes)
        self._print_examples(cer, labels, source, substituted, indexes)
