import pickle
from typing import List

from anonymization.base import Anonymization


class DonatedDatasetAnonymization(Anonymization):
    def __init__(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                 source_text_list: List[List[str]], other_label: str = 'O', **kwargs):
        super().__init__(other_label)
        self.ref_book = dict()
        self.general_ref_book = dict()
        for gen_categories, spec_categories, source_text \
                in zip(specific_category_list, general_category_list, source_text_list):
            for gen_category, spec_category, segment in zip(gen_categories, spec_categories, source_text):
                if gen_category == other_label:
                    continue
                if gen_category not in self.general_ref_book:
                    self.general_ref_book[gen_category] = []
                self.general_ref_book[gen_category].append(segment)
                if spec_category not in self.ref_book:
                    self.ref_book[spec_category] = []
                self.ref_book[spec_category].append(segment)

    @staticmethod
    def _use_saved_dataset_as_donor(path_to_dataset: str, other_label: str = 'O'):
        """
        :param path_to_dataset: путь до сохранённого обработанного набора данных для NER (см. NerDataset)
        :param other_label: метка для незащищаемой сущности
        """
        with open(path_to_dataset, 'rb') as f:
            source_text_list, specific_category_list, general_category_list = pickle.load(f)

        return DonatedDatasetAnonymization(general_category_list, specific_category_list, source_text_list, other_label)

    def _get_substitutions(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                           source_text_list: List[List[str]]) -> List[List[str]]:
        doc_substitutions = []
        for general_categories, specific_categories, entities \
                in zip(general_category_list, specific_category_list, source_text_list):
            substitutions = []
            for general_category, specific_category, entity in zip(general_categories, specific_categories, entities):
                if general_category == self.other_label:
                    continue
                substitution = ''
                if specific_category in self.ref_book:
                    substitution = self.ref_book[specific_category]
                elif general_category in self.general_ref_book:
                    substitution = self.ref_book[specific_category]
                substitutions.append(substitution)
            doc_substitutions.append(substitutions)

        return doc_substitutions
