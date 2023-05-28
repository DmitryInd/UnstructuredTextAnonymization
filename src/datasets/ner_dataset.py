import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

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
            'LOCATION', 'HOSPITAL', 'ORGANIZATION', 'STREET',
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
        ['EMAIL', 'FAX', 'PHONE', 'CONTACT', 'IPADDR', 'IPADDRESS', 'URL']
    ),
    ('O', ['O'])
]


class XMLDataset(Dataset):
    def __init__(self, path_to_folder: str, xml_type="2006", anonymization=None,
                 label_aliases: List[Tuple[str, List[str]]] = None,
                 is_uncased=False, pretrained_tokenizer: str = None, max_length=100,
                 device="cuda:0"):
        """
        :param path_to_folder: путь к директории с xml файлами, содержащими размеченные данные
        :param xml_type: тип набора данных, который будет считываться
        :param anonymization: класс для обезличивания защищённых данных (None -> без обезличивания)
        :param label_aliases: упорядоченный список пар меток и их возможных псевдонимов
        :param is_uncased: приводить все символы к нижнему регистру или нет
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_length: максимальное количество токенов в примере
        :param device: устройство, на котором будет исполняться запрос
        """
        path = Path(path_to_folder)
        assert path.exists(), "The specified folder doesn't exist"
        # Setting up entity labels
        if label_aliases is None:
            label_aliases = LABEL_MEMBERSHIP
        self.alias2label = {}
        for standard, variants in label_aliases:
            for variant in variants:
                self.alias2label[variant] = standard
        self.index2label = list(list(zip(*label_aliases))[0])
        self.label2index = {label: i for i, label in enumerate(self.index2label)}
        # Start of reading files
        self.anonymization = anonymization if anonymization is not None else lambda _1, _2, x: x
        self.is_uncased = is_uncased
        self._record_ids = []
        tokenized_source_list = []
        tokenized_target_list = []
        for xml_file in path.glob("*.xml"):
            read_xml = None
            if xml_type == "2006":
                read_xml = self._read_2006_xml
            ids_batch, source_batch, target_batch = read_xml(str(xml_file))
            self._record_ids.extend(ids_batch)
            tokenized_source_list.extend(source_batch)
            for labels in target_batch:
                tokenized_target_list.append([self.label2index[label] for label in labels])
        # Data tokenization
        self.tokenizer = WordPieceTokenizer(tokenized_source_list,
                                            self.label2index['O'],
                                            True,
                                            max_sent_len=max_length,
                                            pretrained_name=pretrained_tokenizer)
        tokenized = [self.tokenizer(s, t) for s, t in zip(tokenized_source_list, tokenized_target_list)]
        self._tokenized_source_list, self._tokenized_target_list = map(list, zip(*tokenized))
        self.record2idx = {record_id: i for i, record_id in enumerate(self._record_ids)}
        self._record_ids = torch.tensor(self._record_ids, device=device)
        self.device = device

    def _read_2006_xml(self, path_to_file: str) -> Tuple[List[int], List[List[str]], List[List[str]]]:
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
            text = record.find('TEXT')
            for child in text.iter():
                if child.tag == 'PHI':
                    label = self.alias2label.get(child.get("TYPE"), 'O')
                    source_words.append(self.anonymization(label, child.get("TYPE"), child.text))
                    target_labels.append(label)
                    if child.tail is not None:
                        source_words.append(child.tail)
                        target_labels.append('O')
                elif child.tag == 'TEXT':
                    if child.text is not None:
                        source_words.append(child.text)
                        target_labels.append('O')
                elif child.text is not None or child.tail is not None:
                    source_words.append((child.text if child.text is not None else '') + ' ' +
                                        (child.tail if child.tail is not None else ''))
                    target_labels.append('O')

            if self.is_uncased:
                source_words = [s_part.lower() for s_part in source_words]
            record_ids.append(record_id)
            source_batch.append(source_words)
            target_batch.append(target_labels)

        return record_ids, source_batch, target_batch

    def __len__(self):
        return len(self._tokenized_source_list)

    def __getitem__(self, idx):
        source_ids = torch.tensor(self._tokenized_source_list[idx], device=self.device)
        target_ids = torch.tensor(self._tokenized_target_list[idx], device=self.device)

        return self._record_ids[idx], source_ids, target_ids