import pickle
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from anonymization.base import Anonymization
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


class NerDataset(Dataset, ABC):
    def __init__(self, path_to_folder: str, anonymization: Optional[Anonymization] = None,
                 label_aliases: List[Tuple[str, List[str]]] = None, other_label='O',
                 is_uncased=False, pretrained_tokenizer: str = None,
                 max_token_number=100, overlap=40, eq_max_padding=True,
                 cashed_data_path: str = None, device="cpu", **kwargs):
        """
        :param path_to_folder: путь к директории с xml файлами, содержащими размеченные данные /
                               путь к .pkl файлу, в котором данные приведены к нужному формату
        :param anonymization: класс для обезличивания защищённых данных (None -> без обезличивания)
        :param label_aliases: упорядоченный список пар меток и их возможных псевдонимов
        :param other_label: метка для незащищаемой сущности
        :param is_uncased: приводить все символы к нижнему регистру или нет
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_token_number: максимальное количество токенов в примере
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до максимальной указанной длины для всех примеров или
                               паддинг до самого длинного примера в батче
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
        self.other_label = other_label
        self.pad_label = "[PAD]"
        self.index2label = list(list(zip(*label_aliases))[0]) + [self.pad_label]
        self.label2index = {label: i for i, label in enumerate(self.index2label)}
        # Start of reading files
        self.is_uncased = is_uncased
        self.eq_max_padding = eq_max_padding
        self.cashed_data_path = cashed_data_path
        # ([id документа/записи, ...], [List[подряд идущих слов с одной меткой], ...],
        #  [List[Конкретная категория], ...], [List[Общая категория], ...], [List[id категории], ...])
        if path.suffix != '.pkl':
            (record_ids, tokenized_source_list,
             specific_category_list, general_category_list, tokenized_target_list) = self._read_data(path)
            cashed_data_path = path.with_suffix(".pkl")
            with open(str(cashed_data_path), 'wb') as f:
                pickle.dump((record_ids, tokenized_source_list,
                             specific_category_list, general_category_list, tokenized_target_list), f)
        else:
            with open(str(path), 'rb') as f:
                (record_ids, tokenized_source_list,
                 specific_category_list, general_category_list, tokenized_target_list) = pickle.load(f)

        if anonymization is not None:
            tokenized_source_list = anonymization(general_category_list, specific_category_list, tokenized_source_list)
        # Data tokenization
        self.tokenizer = WordPieceTokenizer(tokenized_source_list,
                                            self.label2index[self.pad_label], pad_flag=eq_max_padding,
                                            max_sent_len=max_token_number, overlap=overlap,
                                            pretrained_name=pretrained_tokenizer)
        self._record_ids, self._tokenized_source_list, self._tokenized_target_list = [], [], []
        for record_id, sentence, labels in zip(record_ids, tokenized_source_list, tokenized_target_list):
            tokenized = self.tokenizer(sentence, labels)
            self._record_ids.extend([f"{record_id}:{-i}" for i in range(len(tokenized[0]) - 1, -1, -1)])
            self._tokenized_source_list.extend(tokenized[0])
            self._tokenized_target_list.extend(tokenized[1])
        self.record2idx = {record_id: i for i, record_id in enumerate(self._record_ids)}
        self._record_ids = np.array(self._record_ids)
        self.device = device

    def _read_data(self, path: Path):
        """
        Возвращает из указанной директории данные в формате
        ([id документа/записи, ...], [List[подряд идущих слов с одной меткой], ...],
        [List[Конкретная категория], ...], [List[Общая категория], ...], [List[id категории], ...])
        """
        # For anonymization
        specific_category_list = []
        general_category_list = []
        # Dataset values
        record_ids = []
        tokenized_source_list = []
        tokenized_target_list = []
        for xml_file in path.glob("*." + self.file_extension):
            ids_batch, source_batch, target_batch = self._read_file(str(xml_file))
            record_ids.extend(ids_batch)
            tokenized_source_list.extend(source_batch)
            specific_category_list.extend(target_batch)
            for aliases in target_batch:
                general_category_list.append([self.alias2label.get(alias, self.other_label) for alias in aliases])
                tokenized_target_list.append([self.label2index[label] for label in general_category_list[-1]])
        return record_ids, tokenized_source_list, specific_category_list, general_category_list, tokenized_target_list

    @property
    @abstractmethod
    def file_extension(self) -> str:
        return ""

    @abstractmethod
    def _read_file(self, path_to_file: str) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        """
        :param path_to_file: путь до файла в формате строки
        :return: ([id документа/записи, ...], [List[подряд идущих слов с одной меткой], ...], [List[Метка], ...])
        """
        pass

    def __len__(self):
        return len(self._tokenized_source_list)

    def __getitem__(self, idx):
        source_ids = torch.tensor(self._tokenized_source_list[idx], device=self.device)
        target_ids = torch.tensor(self._tokenized_target_list[idx], device=self.device)

        return self._record_ids[idx], source_ids, target_ids

    def get_collate_fn(self):
        if self.eq_max_padding:
            return None

        token_pad_id = self.tokenizer.word2index[self.tokenizer.pad_token]
        label_pad_id = self.label2index[self.pad_label]

        def collate_fn(sample_list):
            max_len = max(map(len, list(zip(*sample_list))[1]))
            record_ids, batch_token_ids, batch_label_ids = [], [], []
            for record_id, token_ids, label_ids in sample_list:
                record_ids.append(np.expand_dims(record_id, 0))
                filler = torch.ones(max_len - len(token_ids), dtype=torch.long, device=self.device)
                batch_token_ids.append(torch.cat((token_ids, filler * token_pad_id)).unsqueeze(0))
                batch_label_ids.append(torch.cat((label_ids, filler * label_pad_id)).unsqueeze(0))
            return np.concatenate(record_ids), torch.cat(batch_token_ids), torch.cat(batch_label_ids)

        return collate_fn


class I2b2SixNerDataset(NerDataset):
    @property
    def file_extension(self) -> str:
        return "xml"

    def _read_file(self, path_to_file: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        tree = ET.ElementTree(file=path_to_file)
        root = tree.getroot()
        records = root.findall('RECORD')
        record_ids = []
        source_batch = []
        target_batch = []
        for record in records:
            source_words = []
            target_labels = []
            record_id = record.get('ID')
            text = record.find('TEXT')
            for child in text.iter():
                if child.tag == 'PHI':
                    source_words.append(child.text)
                    target_labels.append(child.get("TYPE"))
                    if child.tail is not None:
                        source_words.append(child.tail)
                        target_labels.append(self.other_label)
                elif child.tag == 'TEXT':
                    if child.text is not None:
                        source_words.append(child.text)
                        target_labels.append(self.other_label)
                elif child.text is not None or child.tail is not None:
                    source_words.append((child.text if child.text is not None else '') + ' ' +
                                        (child.tail if child.tail is not None else ''))
                    target_labels.append(self.other_label)

            if self.is_uncased:
                source_words = [s_part.lower() for s_part in source_words]
            record_ids.append(record_id)
            source_batch.append(source_words)
            target_batch.append(target_labels)

        return record_ids, source_batch, target_batch


class I2b2FourteenNerDataset(NerDataset):
    @property
    def file_extension(self) -> str:
        return "xml"

    def _read_file(self, path_to_file: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        tree = ET.ElementTree(file=path_to_file)
        root = tree.getroot()
        record_ids = [Path(path_to_file).stem]
        source_words = []
        target_labels = []
        text = root.find('TEXT').text
        tags = root.find('TAGS')
        current_pos = 0
        for child in tags:
            start = int(child.get("start"))
            if start > current_pos:
                source_words.append(text[current_pos:start])
                target_labels.append(self.other_label)
            source_words.append(child.get("text"))
            target_labels.append(self._phi_type_conversion(child.get("TYPE")))
            current_pos = int(child.get("end"))
        if current_pos < len(text):
            source_words.append(text[current_pos:])
            target_labels.append(self.other_label)
        if self.is_uncased:
            source_words = [s_part.lower() for s_part in source_words]
        return record_ids, [source_words], [target_labels]

    @staticmethod
    def _phi_type_conversion(phi_type):
        if phi_type == "DATE":
            phi_type = "DATEYEAR"
        return phi_type


def get_ner_dataset(dataset_type: str, *args, **kwargs) -> Optional[NerDataset]:
    if dataset_type == "i2b2six":
        return I2b2SixNerDataset(*args, **kwargs)
    elif dataset_type == "i2b2fourteen":
        return I2b2FourteenNerDataset(*args, **kwargs)
    return None
