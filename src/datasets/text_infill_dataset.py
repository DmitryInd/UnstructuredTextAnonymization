import os
import pickle
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
from enum import Enum
from tqdm import tqdm
from collections import Counter
from mask.util import masked_spans_bounds_valid, masked_spans_overlap

from mask.n_gram import NgramsMaskFn, MaskNgramType

RAW_DATA_DIR = str(Path(__file__).absolute().parents[2] / Path('data') / Path('token'))


class DatasetType(Enum):
    CUSTOM = 0
    ARXIV_CS_ABSTRACTS = 1
    ROC_STORIES = 2
    ROC_STORIES_NO_TITLE = 3
    I2B2SIX = 4
    I2B2FOURTEEN = 5


def get_dataset(dataset, split, *args, data_dir=None, shuffle=False, limit=None, **kwargs):
    if not isinstance(dataset, DatasetType):
        raise ValueError('Must specify a Dataset enum value')

    if dataset == DatasetType.CUSTOM:
        d = custom(split, data_dir)
        if data_dir is None:
            raise ValueError('Data dir must be specified for custom dataset')
    elif dataset == DatasetType.ARXIV_CS_ABSTRACTS:
        d = arxiv_cs_abstracts(split, *args, data_dir=data_dir, **kwargs)
    elif dataset == DatasetType.ROC_STORIES:
        d = roc_stories(split, *args, data_dir=data_dir, **kwargs)
    elif dataset == DatasetType.ROC_STORIES_NO_TITLE:
        d = roc_stories(split, *args, data_dir=data_dir, with_titles=False, **kwargs)
    else:
        assert False

    if shuffle:
        random.shuffle(d)

    if limit is not None:
        d = d[:limit]

    return d


class TextInfillDataset(Dataset, ABC):
    def __init__(self, path_to_data: str, split: str = None, max_num_documents=0, is_uncased=False,
                 pretrained_tokenizer: str = None, max_length=100, overlap=40, eq_max_padding=True,
                 device="cuda:0"):
        """
        :param path_to_data: путь к директории с данными, которые необходимо предварительно замаскировать, или к размеченному '.pkl' файлу
        :param split: тип среза данных: train, val, test
        :param max_num_documents: максимальное количество документов
        :param is_uncased: приводить все символы к нижнему регистру или нет
        :param pretrained_tokenizer: путь к сохранённым параметрам токенизатора
        :param max_length: максимальное количество токенов в примере
        :param overlap: пересечение последовательных частей предложений
        :param eq_max_padding: паддинг до максимальной указанной длины для всех примеров или
                               паддинг до самого длинного примера в батче
        :param device: устройство, на котором будет исполняться запрос
        """
        path = Path(path_to_data)
        assert path.exists(), "The specified folder or file doesn't exist"
        # Setting up entity labels
        self.pad_label = "[PAD]"
        # Start of reading files
        self.split = split
        self.max_num_documents = max_num_documents
        self.is_uncased = is_uncased
        self.eq_max_padding = eq_max_padding
        if Path(path_to_data).suffix != '.pkl':
            masked_data = self._read_data(path_to_data)
            path_to_data = str(Path(path_to_data).parent / Path(f'{Path(path_to_data).stem}.pkl'))
            with open(path_to_data, 'wb') as f:
                pickle.dump(masked_data, f)
        # [(текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...]
        dataset = pickle.load(f)

        # Data tokenization
        self.tokenizer = WordPieceTokenizer(tokenized_source_list,
                                            self.label2index[self.pad_label], pad_flag=eq_max_padding,
                                            max_sent_len=max_length, overlap=overlap,
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

    @abstractmethod
    @property
    def _mask_types(self) -> List[Enum]:
        pass

    @abstractmethod
    def _read_data(self, path_to_data: str) -> Tuple[List[int], List[List[str]], List[List[str]]]:
        """
        Считывает данные и возвращает их в формате: текст документа и список наборов масок для него
        :param path_to_data: путь до директории с данными в формате строки
        :return:  [(текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...]
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


class RandomMaskTextInfillDataset(TextInfillDataset):
    def __init__(self, path_to_data: str, split: str = None, max_num_documents=0, is_uncased=False,
                 mask_p=None, max_span_len=None,
                 num_examples=3, max_num_retries=3, min_masked_spans=None, max_masked_spans=None,
                 pretrained_tokenizer: str = None, max_length=100, overlap=40, eq_max_padding=True,
                 device="cuda:0"):
        self.masker = NgramsMaskFn(mask_p, max_span_len)
        self.num_examples = num_examples  # per document
        self.max_num_retries = max_num_retries  # per example
        self.min_masked_spans = min_masked_spans  # per example
        self.max_masked_spans = max_masked_spans  # per example
        super().__init__(path_to_data, split, max_num_documents, is_uncased, pretrained_tokenizer, max_length, overlap,
                         eq_max_padding, device)

    @property
    def _mask_types(self) -> List[Enum]:
        return self.masker.mask_types()

    def _read_data(self, path_to_data: str) -> List[Tuple[str, List[List[Tuple[MaskNgramType, int, int]]]]]:
        docs = self._read_docs(path_to_data)
        masked_data, error_to_count = self.randomly_mask_dataset(docs,
                                                                 random_sample_down_to_max=True,
                                                                 ensure_valid_bounds_in_spans=True,
                                                                 ensure_nonoverlapping_spans=True,
                                                                 ensure_unique=True)
        print(error_to_count)
        return masked_data

    @abstractmethod
    def _read_docs(self, path_to_data: str) -> List[str]:
        pass

    def randomly_mask_dataset(self, docs, **kwargs) \
            -> Tuple[List[Tuple[str, List[List[Tuple[MaskNgramType, int, int]]]]], Counter]:
        """
        :return:  ([(текст документа, список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...], счётчик ошибок при создании масок)
        """
        docs_masked = []
        error_to_count_total = Counter()
        for doc in tqdm(docs):
            doc_masks, error_to_count = self.randomly_mask_document(doc, **kwargs)
            docs_masked.append((doc, doc_masks))
            for k, v in error_to_count.items():
                error_to_count_total[k] += v

        return docs_masked, error_to_count_total

    def randomly_mask_document(self, doc,
                               random_sample_down_to_max=True,
                               ensure_valid_bounds_in_spans=True,
                               ensure_nonoverlapping_spans=True,
                               ensure_unique=True) -> Tuple[List[List[Tuple[MaskNgramType, int, int]]], Counter]:
        """
        :return: (список наборов масок для одного документа: [[(тип, сдвиг, длина), ...], ...], счётчик ошибок при их создании масок)
        """
        error_to_count = Counter()
        doc_masks = []
        doc_masks_set = set()

        def mask_acceptable(masked_spans):
            if self.min_masked_spans is not None and len(masked_spans) < self.min_masked_spans:
                return False, 'Too few spans'

            if self.max_masked_spans is not None and len(masked_spans) > self.max_masked_spans:
                return False, 'Too many spans'

            if ensure_valid_bounds_in_spans and not masked_spans_bounds_valid(masked_spans, len(doc)):
                return False, 'Masked span boundaries are invalid'

            if ensure_nonoverlapping_spans and masked_spans_overlap(masked_spans):
                return False, 'Masked spans overlap'

            if ensure_unique and masked_spans in doc_masks_set:
                return False, 'Mask is not unique'

            return True, None

        for i in range(self.num_examples):
            mask = None
            num_retries = 0
            while num_retries < self.max_num_retries and mask is None:
                try:
                    mask = tuple(self.masker.mask(doc))
                except Exception as e:
                    error_to_count['Mask function exception: {}'.format(str(e))] += 1
                    mask = None

                if mask is not None:
                    if (self.max_masked_spans is not None and random_sample_down_to_max
                            and len(mask) > self.max_masked_spans):
                        mask = tuple(random.sample(mask, self.max_masked_spans))
                    mask_is_acceptable, error_msg = mask_acceptable(mask)
                    if not mask_is_acceptable:
                        error_to_count['Issue with example: {}'.format(error_msg)] += 1
                        mask = None

                num_retries += 1

            if mask is not None:
                doc_masks.append(mask)
                doc_masks_set.add(mask)

        return [list(m) for m in doc_masks], error_to_count


def custom(split, data_dir):
    fp = os.path.join(data_dir, '{}.txt'.format(split))
    try:
        with open(fp, 'r') as f:
            entries = [e.strip() for e in f.read().strip().split('\n\n\n')]
    except:
        raise ValueError('Could not load from {}'.format(fp))
    return entries


ABS_DIR = os.path.join(RAW_DATA_DIR, 'arxiv_cs_abstracts')


def arxiv_cs_abstracts(split='train', data_dir=None, attrs=None):
    if attrs is None:
        attrs = ['title', 'authors', 'categories', 'abstract']
    assert split in ['train', 'valid', 'test']

    if data_dir is None:
        data_dir = ABS_DIR

    with open(os.path.join(data_dir, 'arxiv_cs_abstracts.txt'), 'r') as f:
        raw = f.read().split('\n\n\n')

    abstracts = []
    for r in raw:
        aid, created, updated, categories, title, authors, abstract = r.split('\n', 6)

        a = []
        for attr_name in attrs:
            a.append(eval(attr_name))
        a = '\n'.join(a)

        if created.startswith('2018'):
            if split == 'valid':
                abstracts.append(a)
        elif created.startswith('2019'):
            if split == 'test':
                abstracts.append(a)
        else:
            if split == 'train':
                abstracts.append(a)

    return abstracts


ROC_STORIES_DIR = os.path.join(RAW_DATA_DIR, 'roc_stories')


def roc_stories(split='train', data_dir=None, with_titles=True, exclude_nonstandard=True):
    assert split in ['train', 'valid', 'test', 'test_hand_title']

    if data_dir is None:
        data_dir = ROC_STORIES_DIR

    titled = None
    if split == 'train':
        with open(os.path.join(data_dir, 'train_title.txt'), 'r') as f:
            stories = f.read().split('\n\n\n')
        titled = True
    elif split == 'valid':
        with open(os.path.join(data_dir, 'valid.txt'), 'r') as f:
            stories = f.read().split('\n\n\n')
        titled = False
    elif split == 'test':
        with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
            stories = f.read().split('\n\n\n')
        titled = False
    elif split == 'test_hand_title':
        with open(os.path.join(data_dir, 'test_hand_title.txt'), 'r') as f:
            stories = f.read().split('\n\n\n')
        titled = True

    stories = [s.strip() for s in stories if len(s.strip()) > 0]

    if with_titles != titled:
        if with_titles:
            stories = ['Unknown Title\n{}'.format(s) for s in stories]
        else:
            stories = [s.splitlines()[-1] for s in stories]

    if exclude_nonstandard:
        from nltk.tokenize import sent_tokenize

        standardized = []
        for s in stories:
            paragraphs = s.splitlines()
            if len(paragraphs) != (2 if with_titles else 1):
                continue
            try:
                if len(sent_tokenize(paragraphs[-1])) != 5:
                    continue
            except:
                raise Exception('Need to call nltk.download(\'punkt\')')
            standardized.append(s)
        stories = standardized

    return stories
