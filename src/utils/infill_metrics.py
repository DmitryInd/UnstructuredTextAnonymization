import itertools
import random
from typing import List, Tuple

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
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
        self.is_uncased = is_uncased

        self.anonymization = anonymization
        self.other_label = self.anonymization.other_label
        self.completed_sentences = self.anonymization(general_category_list,
                                                      specific_category_list,
                                                      source_text_list)

        self.general_category_list = sum([[x] * anonymization.var_num for x in general_category_list], [])
        self.specific_category_list = sum([[x] * anonymization.var_num for x in specific_category_list], [])
        self.source_text_list = sum([[x] * anonymization.var_num for x in source_text_list], [])

        self.lemmatizer = WordNetLemmatizer()
        try:
            self.lemmatizer.lemmatize("Lemmatization check")
        except:
            nltk.download('wordnet')
        self.orig_label_lemmas = dict()
        self.sub_label_lemmas = dict()
        self.orig_entity_term_freq = dict()
        self.sub_entity_term_freq = dict()
        self.error_rates = self._calculate_cer()
        self.ex_cer = np.array(list(map(np.mean, self.error_rates)))
        self.avg_cer = self.ex_cer.mean()

    def _calculate_cer(self) -> List[List[float]]:
        cer = []
        # Sub - substituted
        for labels, doc, sub_doc in zip(self.general_category_list, self.source_text_list, self.completed_sentences):
            doc_cer = []
            for label, section, sub_section in zip(labels, doc, sub_doc):
                if label == self.other_label:
                    continue

                section, sub_section = (section.strip().lower(), sub_section.strip().lower()) if self.is_uncased else \
                                       (section.strip(), sub_section.strip())
                doc_cer.append(
                    char_error_rate(sub_section, section).item()
                )
                self._add_lemmas(self.orig_label_lemmas, label, section)
                self._add_lemmas(self.sub_label_lemmas, label, sub_section)
                self._update_entity_counter(self.orig_entity_term_freq, label, section)
                self._update_entity_counter(self.sub_entity_term_freq, label, sub_section)

            cer.append(doc_cer)

        return cer

    def _add_lemmas(self, lemmas_set, label, section):
        if label not in lemmas_set:
            lemmas_set[label] = set()
        lemmas_set[label] |= set(self.lemmatizer.lemmatize(word) for word in section.strip().split())

    @staticmethod
    def _update_entity_counter(entity_counter, label, sub_section):
        if label not in entity_counter:
            entity_counter[label] = dict()
        entity_counter[label][sub_section] = entity_counter[label].get(sub_section, 0) + 1

    def examples_by_indexes(self, indexes) \
            -> Tuple[List[List[float]], List[List[str]], List[List[str]], List[List[str]]]:
        source = [self.source_text_list[i] for i in indexes]
        substituted = [self.completed_sentences[i] for i in indexes]
        labels = [self.general_category_list[i] for i in indexes]
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
                        indexes=None, max_example_len=None, start_other_len=None):
        if max_example_len is not None:
            start_other_len = start_other_len if start_other_len is not None else max_example_len // 2

        for i in range(len(source_texts)):
            labels_set = []
            sources_set = []
            substitutions_set = []
            error_rates_set = []

            cur_labels = []
            cur_source = []
            cur_sub = []
            aligned_error_rate = []

            source_tail = ""
            sub_tail = ""

            cursor = 0
            ex_len = 0
            for j, label in enumerate(labels[i]):
                if label != self.other_label and cursor < len(error_rate[i]):
                    cur_labels.append(label)
                    cur_source.append(self._prepare_text(source_texts[i][j]))
                    cur_sub.append(self._prepare_text(substituted[i][j]))
                    aligned_error_rate.append(f'{error_rate[i][cursor]:.3f}')
                    cursor += 1
                else:
                    aligned_error_rate.append('')
                    cur_labels.append(label)
                    if max_example_len is not None:
                        # Обрезаем до максимального количества символов
                        left_len = max_example_len - ex_len
                        cur_source.append(self._prepare_text(source_texts[i][j][:left_len]))
                        cur_sub.append(self._prepare_text(substituted[i][j][:left_len]))
                        if left_len < len(source_texts[i][j]):
                            # Для следующего примера
                            tail_start = max(len(source_texts[i]) - left_len, start_other_len)
                            source_tail = self._prepare_text(source_texts[i][j][-tail_start:])
                            sub_tail = self._prepare_text(substituted[i][j][-tail_start:])
                    else:
                        cur_source.append(self._prepare_text(source_texts[i][j]))
                        cur_sub.append(self._prepare_text(substituted[i][j]))

                ex_len += len(cur_source[-1])
                if ex_len >= max_example_len:
                    labels_set.append(cur_labels)
                    sources_set.append(cur_source)
                    substitutions_set.append(cur_sub)
                    error_rates_set.append(aligned_error_rate)

                    cur_labels = [self.other_label] if source_tail else []
                    cur_source = [source_tail] if source_tail else []
                    cur_sub = [sub_tail] if source_tail else []
                    aligned_error_rate = [''] if source_tail else []
                    ex_len = len(source_tail)

            if ''.join(aligned_error_rate):
                labels_set.append(cur_labels)
                sources_set.append(cur_source)
                substitutions_set.append(cur_sub)
                error_rates_set.append(aligned_error_rate)

            print('_' * 5 + f' Record {i if indexes is None else indexes[i]} ' + '_' * 5)
            for j in range(len(sources_set)):
                print(tabulate(
                    [
                        ['Labels:'] + labels_set[j],
                        ['Source text:'] + sources_set[j],
                        ['Substituted text:'] + substitutions_set[j],
                        ['CER'] + error_rates_set[j]
                    ],
                    tablefmt='orgtbl'))

    @staticmethod
    def _prepare_text(text: str):
        return ' '.join(text.split('\n'))

    def print_examples_by_indexes(self, indexes, max_example_len=None, start_other_len=None):
        cer, labels, source, substituted = self.examples_by_indexes(indexes)
        self._print_examples(cer, labels, source, substituted, indexes,
                             max_example_len=max_example_len, start_other_len=start_other_len)

    def find_closest_substitutions(self, n):
        """
        Возвращает индексы n самых близких замен сущностей и их значение Character Error Rate
        """
        cer = np.array(list(itertools.chain(*self.error_rates)))
        indexes = np.argsort(cer)[:n]
        cer = cer[indexes]
        row_indexes = np.zeros((len(indexes)), dtype=np.int_)
        stop = np.ones((len(indexes)), dtype=np.int_)
        for row in self.error_rates:
            stop[len(row) > indexes] = 0
            indexes -= stop * len(row)
            row_indexes += stop
            if not any(stop):
                break

        return (row_indexes, indexes), cer
