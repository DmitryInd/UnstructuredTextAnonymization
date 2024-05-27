import os
import pickle
import random
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets.ner_dataset import LABEL_MEMBERSHIP
from datasets.text_infill_tokenization import OfficialGPT2TextInfillTokenizer
from mask.base import MaskFn
from mask.n_gram import MaskNgramType
from mask.personal_entity import MaskEntityType

RAW_DATA_DIR = str(Path(__file__).absolute().parents[2] / Path('data') / Path('token'))


class TextInfillDataset(Dataset, ABC):
    def __init__(self, path_to_data: str, split: str = None, max_num_examples=None, is_uncased=False, with_answers=True,
                 pretrained_tokenizer: str = None, max_full_ex_len=1024, max_only_context_len=None, overlap=256,
                 eq_max_padding=True, device="cuda:0", **kwargs):
        """
        :param path_to_data: путь к директории с данными, которые необходимо предварительно замаскировать, или к размеченному '.pkl' файлу
        :param split: тип среза данных: train, val, test
        :param max_num_examples: максимальное количество документов
        :param is_uncased: приводить ли все символы к нижнему регистру или нет
        :param with_answers: добавлять ли к замаскированному запросу ответы (false, если в данных нет ответов)
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до max_full_ex_len или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        path = Path(path_to_data)
        assert path.exists(), "The specified folder or file doesn't exist"
        # Start of reading files
        self.split = split
        self.max_num_examples = max_num_examples
        self.is_uncased = is_uncased
        self.eq_max_padding = eq_max_padding
        # [(текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...]
        if Path(path_to_data).suffix != '.pkl':
            dataset = self._read_data(path_to_data)
            path_to_data = str(Path(path_to_data).parent / Path(f'{Path(path_to_data).stem}_{split}.pkl'))
            with open(path_to_data, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open(path_to_data, 'rb') as f:
                dataset = pickle.load(f)
        # Data tokenization
        self.tokenizer = OfficialGPT2TextInfillTokenizer(
            pretrained_tokenizer, self._mask_types,
            max_full_ex_len=max_full_ex_len, max_only_context_len=max_only_context_len,
            overlap=overlap, pad_flag=eq_max_padding
        )

        print("Start data tokenization")
        self._record_ids, self._tokenized_source_list, self._tokenized_target_list = [], [], []
        for record_id, doc, mask_sets in tqdm(dataset):
            if self.is_uncased:
                doc = doc.lower()
            token_segments, label_segments = self.tokenizer(doc, mask_sets, with_answers)
            if doc and not token_segments:
                raise ValueError("Tokenization error")
            # For saving order of subsequences
            self._record_ids.extend([f"{record_id}:{i}" for i in range(0, len(token_segments))])
            self._tokenized_source_list.extend(token_segments)
            self._tokenized_target_list.extend(label_segments)

        if self.max_num_examples is not None and self.max_num_examples < len(self._tokenized_source_list):
            example_ids = random.sample(list(range(len(self._tokenized_source_list))), max_num_examples)
            self._record_ids = [self._record_ids[x] for x in example_ids]
            self._tokenized_source_list = [self._tokenized_source_list[x] for x in example_ids]
            self._tokenized_target_list = [self._tokenized_target_list[x] for x in example_ids]

        self.record2idx = {record_id: i for i, record_id in enumerate(self._record_ids)}
        self._record_ids = np.array(self._record_ids)
        self.device = device

    @property
    @abstractmethod
    def _mask_types(self) -> List[Enum]:
        pass

    @abstractmethod
    def _read_data(self, path_to_data: str) -> List[Tuple[str, str, List[List[Tuple[Enum, int, int]]]]]:
        """
        Считывает данные и возвращает их в формате: текст документа и список наборов масок для него
        :param path_to_data: путь до директории с данными в формате строки
        :return:  [(индекс документа, текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...]
        """
        pass

    def __len__(self):
        return len(self._tokenized_source_list)

    def __getitem__(self, idx):
        source_ids = torch.tensor(self._tokenized_source_list[idx], dtype=torch.long, device=self.device)
        target_ids = torch.tensor(self._tokenized_target_list[idx], dtype=torch.long, device=self.device)

        return self._record_ids[idx], source_ids, target_ids

    def get_collate_fn(self):
        if self.eq_max_padding:
            return None

        def collate_fn(sample_list):
            # sample_list: [(record_id, token_inputs, label_inputs), ...]
            record_ids, batch_token_ids, batch_label_ids = zip(*sample_list)
            batch_token_ids, batch_label_ids = self.tokenizer.align_inputs(batch_token_ids, batch_label_ids)
            return np.vstack(tuple(record_ids)), batch_token_ids, batch_label_ids

        return collate_fn


class MaskedTextInfillDataset(TextInfillDataset):
    def __init__(self, path_to_data: str, masker: MaskFn, split: str = None,
                 max_num_examples=None, is_uncased=False, with_answers=True,
                 pretrained_tokenizer: str = None, max_full_ex_len=1024,  max_only_context_len=None, overlap=256,
                 eq_max_padding=True, device="cuda:0", **kwargs):
        """
        :param path_to_data: путь к директории с данными, которые необходимо предварительно замаскировать,
                             или к размеченному '.pkl' файлу
        :param masker:
        :param split: тип среза данных: train, val, test
        :param max_num_examples: максимальное количество документов
        :param is_uncased: приводить ли все символы к нижнему регистру или нет
        :param with_answers: добавлять ли к замаскированному запросу ответы (false, если в данных нет ответов)
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до max_full_ex_len или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        self.masker = masker
        super().__init__(path_to_data, split, max_num_examples, is_uncased, with_answers, pretrained_tokenizer,
                         max_full_ex_len, max_only_context_len, overlap, eq_max_padding, device)

    @property
    def _mask_types(self) -> List[Enum]:
        return self.masker.mask_types

    def _read_data(self, path_to_data: str) -> List[Tuple[str, str, List[List[Tuple[Enum, int, int]]]]]:
        docs = self._read_docs(path_to_data)
        masked_data, error_to_count = self.randomly_mask_dataset(docs)
        print(f'There were next errors in random masking dataset:\n{error_to_count}')
        masked_data_with_id = [(str(i), t, m) for i, (t, m) in enumerate(masked_data)]
        return masked_data_with_id

    @abstractmethod
    def _read_docs(self, path_to_data: str) -> List[str]:
        pass

    def randomly_mask_dataset(self, docs) \
            -> Tuple[List[Tuple[str, List[List[Tuple[Enum, int, int]]]]], Counter]:
        """
        :return:  ([(текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...],
                     счётчик ошибок при создании масок)
        """
        docs_masked = []
        error_to_count_total = Counter()
        for doc in tqdm(docs):
            doc_masks, error_to_count = self.masker.mask(doc)
            docs_masked.append((doc, doc_masks))
            for k, v in error_to_count.items():
                error_to_count_total[k] += v

        return docs_masked, error_to_count_total


class ArxivMaskedTextInfillDataset(MaskedTextInfillDataset):
    def _read_docs(self, path_to_data: str) -> List[str]:
        attrs = ['title', 'authors', 'categories', 'abstract']
        assert self.split in ['train', 'valid', 'test']

        if path_to_data is None:
            path_to_data = os.path.join(RAW_DATA_DIR, 'arxiv_cs_abstracts')

        with open(os.path.join(path_to_data, 'arxiv_cs_abstracts.txt'), 'r') as f:
            raw = f.read().split('\n\n\n')

        abstracts = []
        for r in raw:
            aid, created, updated, categories, title, authors, abstract = r.split('\n', 6)

            a = []
            for attr_name in attrs:
                a.append(eval(attr_name))
            a = '\n'.join(a)

            if created.startswith('2018'):
                if self.split == 'valid':
                    abstracts.append(a)
            elif created.startswith('2019'):
                if self.split == 'test':
                    abstracts.append(a)
            else:
                if self.split == 'train':
                    abstracts.append(a)

        return abstracts


class StoriesMaskedTextInfillDataset(MaskedTextInfillDataset):
    def __init__(self, path_to_data: str, masker: MaskFn, split: str = None,
                 with_titles=True, exclude_nonstandard=True, max_num_examples=None, is_uncased=False, with_answers=True,
                 pretrained_tokenizer: str = None, max_full_ex_len=1024, max_only_context_len=None, overlap=256,
                 eq_max_padding=True, device="cuda:0", **kwargs):
        """
        :param path_to_data: путь к директории с данными, которые необходимо предварительно замаскировать, или к размеченному '.pkl' файлу
        :param split: тип среза данных: train, val, test
        :param with_titles: добавлять ли в документы названия рассказов
        :param exclude_nonstandard: исключать ли рассказы, не подходящие под стандартный шаблон
        :param max_num_examples: максимальное количество документов
        :param is_uncased: приводить ли все символы к нижнему регистру или нет
        :param with_answers: добавлять ли к замаскированному запросу ответы (false, если в данных нет ответов)
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до max_full_ex_len или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        self.with_titles = with_titles
        self.exclude_nonstandard = exclude_nonstandard
        super().__init__(path_to_data, masker, split, max_num_examples, is_uncased, with_answers,
                         pretrained_tokenizer, max_full_ex_len, max_only_context_len, overlap, eq_max_padding, device)

    def _read_docs(self, path_to_data: str) -> List[str]:
        assert self.split in ['train', 'valid', 'test', 'test_hand_title']

        if path_to_data is None:
            path_to_data = os.path.join(RAW_DATA_DIR, 'roc_stories')

        titled = None
        if self.split == 'train':
            with open(os.path.join(path_to_data, 'train_title.txt'), 'r') as f:
                stories = f.read().split('\n\n\n')
            titled = True
        elif self.split == 'valid':
            with open(os.path.join(path_to_data, 'valid.txt'), 'r') as f:
                stories = f.read().split('\n\n\n')
            titled = False
        elif self.split == 'test':
            with open(os.path.join(path_to_data, 'test.txt'), 'r') as f:
                stories = f.read().split('\n\n\n')
            titled = False
        elif self.split == 'test_hand_title':
            with open(os.path.join(path_to_data, 'test_hand_title.txt'), 'r') as f:
                stories = f.read().split('\n\n\n')
            titled = True

        stories = [s.strip() for s in stories if len(s.strip()) > 0]

        if self.with_titles != titled:
            if self.with_titles:
                stories = ['Unknown Title\n{}'.format(s) for s in stories]
            else:
                stories = [s.splitlines()[-1] for s in stories]

        if self.exclude_nonstandard:
            from nltk.tokenize import sent_tokenize
            import nltk

            standardized = []
            for s in stories:
                paragraphs = s.splitlines()
                if len(paragraphs) != (2 if self.with_titles else 1):
                    continue
                try:
                    if len(sent_tokenize(paragraphs[-1])) != 5:
                        continue
                except:
                    nltk.download('punkt')
                standardized.append(s)
            stories = standardized

        return stories


class MarkedUpTextInfillDataset(TextInfillDataset):
    def __init__(self, path_to_data: str, split: str = None, max_num_examples=None,
                 is_uncased=False, with_answers=True,
                 other_label: str = '0', label2type=None, mask_types: Optional[List[Enum]] = None,
                 pretrained_tokenizer: str = None, max_full_ex_len=768, max_only_context_len=1024, overlap=256,
                 eq_max_padding=True, device="cuda:0", **kwargs):
        """
        :param path_to_data: путь к директории с данными, которые необходимо предварительно замаскировать, или к размеченному '.pkl' файлу
        :param split: тип среза данных: train, val, test
        :param max_num_examples: максимальное количество документов
        :param is_uncased: приводить ли все символы к нижнему регистру или нет
        :param with_answers: добавлять ли к замаскированному запросу ответы (false, если в данных нет ответов)
        :param other_label: метка для неперсональных данных
        :param label2type: функция для конвертации названия метки в enum тип маски
        :param mask_types: отсортированный список всех возможных типов масок
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до max_full_ex_len или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        self.other_label = other_label
        self._get_type = label2type if label2type is not None else get_ngram_type
        self._p_mask_types = mask_types if mask_types is not None else list(MaskNgramType)
        super().__init__(path_to_data, split, max_num_examples, is_uncased, with_answers, pretrained_tokenizer,
                         max_full_ex_len, max_only_context_len, overlap, eq_max_padding, device)

    @property
    def _mask_types(self) -> List[Enum]:
        return self._p_mask_types

    def _read_data(self, path_to_data: str) -> List[Tuple[str, str, List[List[Tuple[Enum, int, int]]]]]:
        docs = self._read_docs(path_to_data)
        masked_data = []
        for doc in zip(*docs):
            masked_data.append(self.convert_dataset(doc))
        return masked_data

    @abstractmethod
    def _read_docs(self, path_to_data: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        pass

    def convert_dataset(self, doc: Tuple[str, List[str], List[str]]) \
            -> Tuple[str, str, List[List[Tuple[Enum, int, int]]]]:
        """
        Переводит представление документа\n
        из формата: (id документа, список слов в документе, список классов/меток слов в документе)\n
        в формат: (id документа, текст документа в формате строки,
        (тип замаскированного объекта, сдвиг на начало замаскированного объекта, длина замаскированного объекта))
        """
        masks = []
        text = ""
        for subseq, label in zip(doc[1], doc[2]):
            if label != self.other_label:
                masks.append((self._get_type(label), len(text), len(subseq)))
            text += subseq
        return doc[0], text, [masks]


class FromListMarkedUpTextInfillDataset(MarkedUpTextInfillDataset):
    def __init__(self, path_to_data: str, marked_up_docs: Tuple[List[str], List[List[str]], List[List[str]]],
                 split: str = None, max_num_examples=None, is_uncased=False, with_answers=True,
                 other_label: str = '0', label2type=None, mask_types: Optional[List[Enum]] = None,
                 pretrained_tokenizer: str = None, max_full_ex_len=1024, max_only_context_len=1024, overlap=256,
                 eq_max_padding=True, device="cuda:0", **kwargs):
        """
        :param path_to_data: фиктивный путь к директории с данными, которые необходимо предварительно замаскировать,
                             можно использовать для кэширования данных в '.pkl' файл
        :param marked_up_docs: размеченные данные в формате (id документа, список слов в документе, список классов/меток слов в документе)
        :param split: тип среза данных: train, val, test
        :param max_num_examples: максимальное количество документов
        :param is_uncased: приводить ли все символы к нижнему регистру или нет
        :param with_answers: добавлять ли к замаскированному запросу ответы (false, если в данных нет ответов)
        :param other_label: метка для неперсональных данных
        :param label2type: функция для конвертации названия метки в enum тип маски
        :param mask_types: отсортированный список всех возможных типов масок
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до max_full_ex_len или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        self.marked_up_docs = marked_up_docs
        super().__init__(path_to_data, split, max_num_examples, is_uncased, with_answers, other_label,
                         label2type, mask_types, pretrained_tokenizer, max_full_ex_len, max_only_context_len, overlap,
                         eq_max_padding, device)

    def _read_docs(self, path_to_data: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        return self.marked_up_docs


class I2b2SixNerDatasetMarkedUpTextInfillDataset(MarkedUpTextInfillDataset):
    def _read_docs(self, path_to_data: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        label_aliases = LABEL_MEMBERSHIP
        alias2label = {}
        for standard, variants in label_aliases:
            for variant in variants:
                alias2label[variant] = standard

        path_to_data = Path(path_to_data).glob("*.xml").__iter__().__next__()
        tree = ET.ElementTree(file=path_to_data)
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
                    target_labels.append(alias2label.get(child.get("TYPE"), self.other_label))
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


class I2b2FourteenNerDatasetMarkedUpTextInfillDataset(MarkedUpTextInfillDataset):
    def _read_docs(self, path_to_data: str) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        label_aliases = LABEL_MEMBERSHIP
        alias2label = {}
        for standard, variants in label_aliases:
            for variant in variants:
                alias2label[variant] = standard

        path = Path(path_to_data)
        record_ids = []
        tokenized_source_list = []
        tokenized_target_list = []
        for xml_file in path.glob("*.xml"):
            ids_batch, source_batch, target_batch = self._read_file(str(xml_file))
            record_ids.append(ids_batch)
            tokenized_source_list.append(source_batch)
            tokenized_target_list.append([alias2label.get(alias, self.other_label) for alias in target_batch])

        return record_ids, tokenized_source_list, tokenized_target_list

    def _read_file(self, path_to_file: str) -> Tuple[str, List[str], List[str]]:
        tree = ET.ElementTree(file=path_to_file)
        root = tree.getroot()
        record_id = Path(path_to_file).stem
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
        return record_id, source_words, target_labels

    @staticmethod
    def _phi_type_conversion(phi_type):
        if phi_type == "DATE":
            phi_type = "DATEYEAR"
        return phi_type


def get_ngram_type(label: str) -> Enum:
    """Независимо от названия метки в формате строки возвращает Ngram тип маски"""
    return MaskNgramType.NGRAM


def get_personal_entity_type(label: str) -> Enum:
    """Из метки [подстроки] в формате строки получает тип маски для именованных сущностей с личной информацией"""
    return MaskEntityType[label.upper()]


class DatasetType(Enum):
    ARXIV_CS_ABSTRACTS = 0
    ROC_STORIES = 1
    ROC_STORIES_NO_TITLE = 2
    I2B2SIX = 3
    I2B2FOURTEEN = 4


def get_text_infill_dataset(dataset_type, *args, **kwargs) -> TextInfillDataset:
    if isinstance(dataset_type, str):
        dataset_type = DatasetType[dataset_type.upper()]

    if dataset_type == DatasetType.ARXIV_CS_ABSTRACTS:
        d = ArxivMaskedTextInfillDataset(*args, **kwargs)
    elif dataset_type == DatasetType.ROC_STORIES:
        d = StoriesMaskedTextInfillDataset(**kwargs)
    elif dataset_type == DatasetType.ROC_STORIES_NO_TITLE:
        kwargs['with_titles'] = False
        d = StoriesMaskedTextInfillDataset(**kwargs)
    elif dataset_type == DatasetType.I2B2SIX:
        d = I2b2SixNerDatasetMarkedUpTextInfillDataset(*args, **kwargs)
    elif dataset_type == DatasetType.I2B2FOURTEEN:
        d = I2b2FourteenNerDatasetMarkedUpTextInfillDataset(*args, **kwargs)
    else:
        assert False, "There is no such type of dataset"

    return d
