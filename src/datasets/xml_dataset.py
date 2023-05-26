import re
from typing import Tuple, List

import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from pathlib import Path

from datasets.tokenization import WordPieceTokenizer

LABEL_MEMBERSHIP = [
    (
        'NAME',
        [
         'NAME',
         'DOCTOR',
         'PATIENT',
         'USERNAME',
         'HCPNAME',
         'RELATIVEPROXYNAME',
         'PTNAME',
         'PTNAMEINITIAL',
         'KEYVALUE',
        ]
    ),
    ('PROFESSION', ['PROFESSION']),
    (
        'LOCATION',
        [
            'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'URL', 'STREET',
            'STATE', 'CITY', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
            'PROTECTED_ENTITY', 'PROTECTED ENTITY', 'NATIONALITY'
        ]
    ),
    ('AGE', ['AGE', 'AGE_>_89', 'AGE > 89']),
    ('DATE', ['DATE', 'DATEYEAR']),
    (
        'ID',
        [
            'BIOID', 'DEVICE', 'HEALTHPLAN', 'IDNUM', 'MEDICALRECORD',
            'ID', 'IDENTIFIER', 'OTHER'
        ]
    ),
    (
        'CONTACT',
        ['EMAIL', 'FAX', 'PHONE', 'CONTACT', 'IPADDR', 'IPADDRESS']
    ),
    ('O', ['O'])
]


class BertDataset(Dataset):
    def __init__(self, path_to_folder: str,
                 labels: List[str],
                 pretrained_tokenizer: str = None, max_length=100,
                 device="cuda:0"):
        """
        :param path_to_folder: путь к директории с xml файлами, содержащими размеченные данные
        :param labels: упорядоченный список меток
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_length: максимальное количество токенов в примере
        :param device: устройство, на котором будет исполняться запрос
        """
        path = Path(path_to_folder)
        assert path.exists(), "Specified folder doesn't exist"
        self.index2label = list(list(zip(*LABEL_MEMBERSHIP))[0])
        self.label2index = {label: i for i, label in enumerate(labels)}
        self.record_ids = []
        tokenized_source_list = []
        tokenized_target_list = []
        for xml_file in path.glob("*.xml"):
            ids_batch, source_batch, target_batch = self._read_xml(str(xml_file))
            self.record_ids.extend(ids_batch)
            tokenized_source_list.extend(source_batch)
            tokenized_target_list.extend([self.label2index[label] for label in target_batch])

        self.tokenizer = WordPieceTokenizer(tokenized_source_list,
                                            self.label2index['O'],
                                            True,
                                            max_sent_len=max_length,
                                            pretrained_name=pretrained_tokenizer)
        tokenized = [self.tokenizer(s, t) for s, t in zip(tokenized_source_list, tokenized_target_list)]
        self.tokenized_source_list, self.tokenized_target_list = map(list, zip(*tokenized))
        self.record_ids = torch.tensor(self.record_ids)
        self.device = device

    @staticmethod
    def _read_xml(path_to_file: str) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        to_standard_labels = {}
        for standard, variants in LABEL_MEMBERSHIP:
            for variant in variants:
                to_standard_labels[variant] = standard

        phi_start = re.compile(r"<PHI TYPE=\"([^\"]+)\">")
        phi_end = re.compile(r"</PHI>")
        tree = ET.ElementTree(file=path_to_file)
        root = tree.getroot()
        records = root.findall('RECORD')
        record_ids = []
        source_batch = []
        target_batch = []
        for record in records:
            source_words = []
            target_labels = []
            record_id = int(record.get('ID'))
            text = record.find('text')
            phis = text.findall('PHI')
            phi_id = 0
            index = 0
            for match in phi_start.finditer(text.text):
                source_words.append(text.text[index: match.start()])
                target_labels.append("OTHER")
                index = phi_end.search(text.text[match.start():]).end()
                source_words.append(phis[phi_id].text)
                target_labels.append(to_standard_labels.get(phis[phi_id].get("TYPE"), 'O'))
                phi_id += 1

            if index != len(text.text):
                source_words.append(text.text[index:])
                target_labels.append("OTHER")

            record_ids.append(record_id)
            source_batch.append(source_words)
            target_batch.append(target_labels)

        return record_ids, source_batch, target_batch

    def __len__(self):
        return len(self.tokenized_source_list)

    def __getitem__(self, idx):
        source_ids = torch.tensor(self.tokenized_source_list[idx]).to(self.device)
        target_ids = torch.tensor(self.tokenized_target_list[idx]).to(self.device)

        return self.record_ids[idx], source_ids, target_ids
