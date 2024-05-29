import string
from datetime import datetime
from typing import List

import numpy as np

from anonymization.base import Anonymization


class ReferenceBookAnonymization(Anonymization):
    def __init__(self,
                 path_to_first_male_names: str,
                 path_to_first_femail_names: str,
                 path_to_last_names: str,
                 path_to_full_addresses: str,
                 path_to_countries: str,
                 path_to_states: str,
                 path_to_cities: str,
                 path_to_streets: str,
                 path_to_organizations: str,
                 path_to_hospitals: str,
                 path_to_professions: str,
                 other_label: str = 'O',
                 **kwargs):
        super().__init__(other_label)
        self.professions = self._read_ref_book(path_to_professions)
        self.hospitals = self._read_ref_book(path_to_hospitals)
        self.organizations = self._read_ref_book(path_to_organizations)
        self.streets = self._read_ref_book(path_to_streets)
        self.cities = self._read_ref_book(path_to_cities)
        self.states = self._read_ref_book(path_to_states)
        self.countries = self._read_ref_book(path_to_countries)
        self.full_addresses = self._read_ref_book(path_to_full_addresses)
        self.path_to_last_names = self._read_ref_book(path_to_last_names)
        self.first_femail_names = self._read_ref_book(path_to_first_femail_names)
        self.first_male_names = self._read_ref_book(path_to_first_male_names)

    @staticmethod
    def _read_ref_book(path_to_file: str):
        with open(path_to_file, "r") as file:
            ref_book_list = (file.read()).split("\n")

        return ref_book_list

    def _get_substitutions(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                           source_text_list: List[List[str]]) -> List[List[str]]:
        doc_substitutions = []
        for general_categories, specific_categories, entities \
                in zip(general_category_list, specific_category_list, source_text_list):
            substitutions = []
            for general_category, specific_category, entity in zip(general_categories, specific_categories, entities):
                if general_category != self.other_label:
                    substitutions.append(self._generate_entity(general_category, specific_category, entity))
            doc_substitutions.append(substitutions)

        return doc_substitutions

    def _generate_entity(self, general_category: str, specific_category: str, entity: str):
        deid_entity = ''
        if general_category == 'ID':
            deid_entity = self.generate_random_id()
        elif general_category == 'AGE':
            deid_entity = self.generate_age("89" in specific_category)
        elif general_category == 'NAME':
            deid_entity = self.generate_name()
        elif general_category == 'PROFESSION':
            deid_entity = self.generate_from_ref_book(self.professions)
        elif general_category == "DATE":
            deid_entity = self.generate_date(with_year=(specific_category == "DATEYEAR"))
        elif general_category == "CONTACT":
            if specific_category == "FAX":
                deid_entity = self.generate_random_id(10)
            elif specific_category == "PHONE" or specific_category == "CONTACT":
                deid_entity = self.generate_phone_number()
            elif specific_category == "EMAIL":
                deid_entity = self.generate_email()
            elif specific_category == "URL":
                deid_entity = self.generate_url()
            elif specific_category == "IPADDR" or specific_category == "IPADDRESS":
                deid_entity = self.generate_random_ip()
            else:
                deid_entity = self.generate_phone_number()
        elif general_category == 'LOCATION':
            if specific_category == 'HOSPITAL':
                deid_entity = self.generate_from_ref_book(self.hospitals)
            elif specific_category == 'ORGANIZATION':
                deid_entity = self.generate_from_ref_book(self.organizations)
            elif specific_category == 'STREET':
                deid_entity = self.generate_from_ref_book(self.streets)
            elif specific_category == 'STATE':
                deid_entity = self.generate_from_ref_book(self.states)
            elif specific_category == 'CITY':
                deid_entity = self.generate_from_ref_book(self.cities)
            elif specific_category == 'COUNTRY' or specific_category == 'NATIONALITY':
                deid_entity = self.generate_from_ref_book(self.countries)
            elif specific_category == 'ZIP':
                deid_entity = self.generate_random_id(5)
            else:
                deid_entity = self.generate_from_ref_book(self.full_addresses)
        return str(deid_entity)

    @staticmethod
    def generate_random_id(digits_number: int = 0):
        if digits_number == 0:
            digits_number = np.random.choice([4, 6, 10, 12, 16])
        min_number = 10 ** (digits_number - 1)
        max_number = (10 ** digits_number) - 1
        return np.random.randint(min_number, max_number)

    @staticmethod
    def generate_age(is_more_89: bool):
        return np.random.randint(0, 90) if not is_more_89 else np.random.randint(90, 120)

    @staticmethod
    def generate_date(date_format=None, with_year=False):
        if date_format is None:
            delim = np.random.choice(['/', '-', ' '])
            if with_year:
                variants = [["%d", "%m", "%Y"], ["%Y", "%m", "%d"], ["%m", "%d", "%Y"]]
            else:
                variants = [["%d", "%m"], ["%m", "%d"]]
            date_format = delim.join(variants[np.random.randint(0, len(variants))])
            date_format = date_format.replace("%m", np.random.choice(["%m"] * 4 + ["%b"] * 2 + ["%B"]))
        year = np.random.randint(1900, 2100)
        month = np.random.choice(range(1, 12), 1)[0]
        if month in [1, 3, 5, 7, 8, 10, 12]:
            day = np.random.choice(range(1, 31), 1)[0]
        elif month == 2:
            day = np.random.choice(range(1, 28), 1)[0]
        else:
            day = np.random.choice(range(1, 30), 1)[0]
        return datetime(year=year, month=month, day=day).strftime(date_format)

    @staticmethod
    def generate_phone_number():
        mode = np.random.randint(0, 1)
        pattern = "({}) {}-{}{}"
        if mode:
            pattern = "{}-{}-{}{}"
        number = pattern.format(ReferenceBookAnonymization.generate_random_id(3),
                                ReferenceBookAnonymization.generate_random_id(3),
                                ReferenceBookAnonymization.generate_random_id(2),
                                ReferenceBookAnonymization.generate_random_id(2))
        return number

    @staticmethod
    def generate_random_ip():
        return "{}.{}.{}.{}".format(*np.random.randint(0, 256, 4))

    @staticmethod
    def generate_from_ref_book(variants: List[str]):
        return np.random.choice(variants)

    def generate_name(self):
        is_male = np.random.randint(0, 1)
        if is_male:
            first = np.random.choice(self.first_male_names)
        else:
            first = np.random.choice(self.first_femail_names)
        last = np.random.choice(self.path_to_last_names)
        return first + " " + last

    def generate_email(self):
        is_male = np.random.randint(0, 1)
        if is_male:
            first = np.random.choice(self.first_male_names)
        else:
            first = np.random.choice(self.first_femail_names)
        last = np.random.choice(self.path_to_last_names)
        first = first.split()[0].lower()
        last = last.split()[0].lower()
        indexes = np.random.randint(0, len(string.ascii_lowercase), np.random.randint(2, 6))
        domain = ''.join([string.ascii_lowercase[i] for i in indexes])
        return "{}.{}@{}.com".format(first, last, domain)

    def generate_url(self):
        connection_prefix = np.random.randint(0, 2)
        is_www = np.random.randint(0, 1)
        prefix = ""
        if connection_prefix:
            prefix += ["https://", "http://"][connection_prefix - 1]
        if is_www:
            prefix += "www"
        is_male = np.random.randint(0, 1)
        if is_male:
            first = np.random.choice(self.first_male_names)
        else:
            first = np.random.choice(self.first_femail_names)
        last = np.random.choice(self.path_to_last_names)
        first = first.split()[0].lower()
        last = last.split()[0].lower()
        indexes = np.random.randint(0, len(string.ascii_lowercase), np.random.randint(2, 6))
        domain = ''.join([string.ascii_lowercase[i] for i in indexes])
        return "{}.{}.{}.{}.com".format(prefix, domain, first, last)
