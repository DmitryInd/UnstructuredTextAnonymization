import random
from collections import Counter
from enum import Enum
from functools import lru_cache
from typing import List, Tuple, Optional

from nltk import sent_tokenize, word_tokenize as nltk_word_tokenize, download as nltk_download

from mask.base import MaskFn
from mask.util import masked_spans_bounds_valid, masked_spans_overlap
from mask.util import tokens_offsets


class MaskNgramType(Enum):
    NGRAM = 0


class RandomNgramsMaskFn(MaskFn):
    def __init__(self, p: float = 0.03, mask_types: Optional[List[Enum]] = None,
                 max_ngram_mask_length: int = 8, num_examples_per_doc: int = 3, max_num_retries_per_ex: int = 3,
                 min_masked_spans_per_ex: Optional[int] = None, max_masked_spans_per_ex: Optional[int] = None,
                 ensure_valid_bounds_in_spans=True, ensure_nonoverlapping_spans=True, ensure_unique=True, **kwargs):
        """
        :param p: вероятность замаскировать n-грамму, начиная с указанного слова
        :param mask_types: список возможных типов масок, все типы присваивается случайно с одинаковой вероятностью
        :param max_ngram_mask_length: максимальная длина замаскированной n-граммы
        :param num_examples_per_doc: количество различных разметок / маскировок одного документа
        :param max_num_retries_per_ex: количество попыток корректно замаскировать документ для одной разметки
        :param min_masked_spans_per_ex: минимальное количество масок в одной разметке документа
        :param max_masked_spans_per_ex: максимальное количество масок в одной разметке документа
        :param ensure_valid_bounds_in_spans: проверять ли корректность границ масок
        :param ensure_nonoverlapping_spans: проверять ли пересечение масок
        :param ensure_unique: проверять ли уникальность каждой маски
        """
        try:
            sent_tokenize('Ensure punkt installed.')
        except:
            nltk_download('punkt')
        self.p = p
        self._p_mask_types = mask_types if mask_types is not None else list(MaskNgramType)
        self.max_ngram_mask_length = max_ngram_mask_length
        self.num_examples_per_doc = num_examples_per_doc  # per document
        self.max_num_retries_per_ex = max_num_retries_per_ex  # per example
        self.min_masked_spans_per_ex = min_masked_spans_per_ex  # per example
        self.max_masked_spans_per_ex = max_masked_spans_per_ex  # per example

        self.ensure_valid_bounds_in_spans = ensure_valid_bounds_in_spans
        self.ensure_nonoverlapping_spans = ensure_nonoverlapping_spans
        self.ensure_unique = ensure_unique

    @property
    def mask_types(self) -> List[Enum]:
        return self._p_mask_types

    def get_one_mask(self, doc) -> List[Tuple[Enum, int, int]]:
        """
        Маскирует случайные n-граммы слов в документе. Длина n выбирается случайным образом равномерно между 1 и
        указанной максимальной длиной.
        :param doc: текст документа в формате строки
        :return: список троек:
                (тип замаскированного объекта, сдвиг на начало замаскированного объекта, длина замаскированного объекта)
        """
        doc_offs = doc_to_hierarchical_offsets(doc)

        def _trial(p):
            if p <= 0:
                return False
            else:
                return random.random() < p

        masked_spans = []

        doc_off, doc_len, p_offs = doc_offs
        for p_off, p_len, s_offs in p_offs:
            for s_off, s_len, w_offs in s_offs:
                w_i = 0
                while w_i < len(w_offs):
                    w_off, w_len = w_offs[w_i]
                    if _trial(self.p):
                        # Mask ngram starting at word
                        max_k = min(len(w_offs) - w_i, self.max_ngram_mask_length)
                        assert max_k > 0
                        k = random.randint(1, max_k)
                        last_w_off, last_w_len = w_offs[w_i + k - 1]
                        masked_spans.append((random.choice(self.mask_types), w_off, last_w_off + last_w_len - w_off))
                        w_i += k
                    else:
                        w_i += 1

        return masked_spans

    def mask(self, doc) -> Tuple[List[List[Tuple[Enum, int, int]]], Counter]:
        """
        Создаёт случайным образом несколько независимых наборов масок для документа
        :return: (список наборов масок для одного документа: [[(тип, сдвиг, длина), ...], ...],
                  счётчик ошибок при их создании масок)
        """
        error_to_count = Counter()
        doc_masks = []
        doc_masks_set = set()

        def mask_acceptable(masked_spans):
            if self.min_masked_spans_per_ex is not None and len(masked_spans) < self.min_masked_spans_per_ex:
                return False, 'Too few spans'

            if self.max_masked_spans_per_ex is not None and len(masked_spans) > self.max_masked_spans_per_ex:
                return False, 'Too many spans'

            if self.ensure_valid_bounds_in_spans and not masked_spans_bounds_valid(masked_spans, len(doc)):
                return False, 'Masked span boundaries are invalid'

            if self.ensure_nonoverlapping_spans and masked_spans_overlap(masked_spans):
                return False, 'Masked spans overlap'

            if self.ensure_unique and set(masked_spans) & doc_masks_set:
                return False, 'Mask is not unique'

            return True, None

        for i in range(self.num_examples_per_doc):
            mask = None
            num_retries = 0
            while num_retries < self.max_num_retries_per_ex and mask is None:
                try:
                    mask = self.get_one_mask(doc)
                except Exception as e:
                    error_to_count['Mask function exception: {}'.format(str(e))] += 1
                    mask = None

                if mask is not None:
                    if self.max_masked_spans_per_ex is not None and len(mask) > self.max_masked_spans_per_ex:
                        m_indexes = sorted(random.sample(list(range(len(mask))), self.max_masked_spans_per_ex))
                        mask = [mask[index] for index in m_indexes]
                    mask_is_acceptable, error_msg = mask_acceptable(mask)
                    if not mask_is_acceptable:
                        error_to_count['Issue with example: {}'.format(error_msg)] += 1
                        mask = None

                num_retries += 1

            if mask is not None:
                doc_masks.append(mask)
                doc_masks_set.update(mask)

        return doc_masks, error_to_count


@lru_cache(maxsize=128)
def doc_to_hierarchical_offsets(d, relative=False):
    """
    Рассчитывает рекурсивные токены (токены внутри токенов).
    :param d: документ в формате строки
    :param relative: считать ли сдвиг от начала документа, или от начала токена на уровень выше
    :return: токены в формате: (сдвиг к началу токена; длина токена; внутренние токены, на которые разбит текущий: (..., ..., ))
    """
    tokenize_fns = [
        # Preserve original doc
        lambda d: [d],
        # Tokenize into paragraphs
        lambda d: [p.strip() for p in d.splitlines() if len(p.strip()) > 0],
        # Tokenize into sentences
        lambda p: [s.strip() for s in sent_tokenize(p) if len(s.strip()) > 0],
        # Tokenize into words
        lambda s: [w.strip() for w in word_tokenize(s) if len(w.strip()) > 0]
    ]
    return _hierarchical_offsets_recursive(d, tokenize_fns, relative=relative)[0]


def word_tokenize(x):
    x_tokens = nltk_word_tokenize(x)
    x_tokens_offsets = tokens_offsets(x, x_tokens)
    for i, off in enumerate(x_tokens_offsets):
        if off is None and '\"' in x and (x_tokens[i] == '``' or x_tokens[i] == '\'\''):
            x_tokens[i] = '\"'
    return x_tokens


def _hierarchical_offsets_recursive(x, tokenize_fns, relative=False, parent_off=0):
    """
    Рассчитывает рекурсивные токены (токены внутри токенов).
    :param x: токен / строка, которая разбивается на токены
    :param tokenize_fns: функции, разбивающие строку на токены, в порядке рекурсии
    :param relative: суммировать ли сдвиг к началу токена с родительским сдвигом
    :param parent_off: сдвиг родительского токена, который будет разбиваться в текущей итерации
    :return: токены в формате: (сдвиг к началу токена; длина токена; внутренние токены, на которые разбит текущий: (..., ..., ), )
    """
    if len(tokenize_fns) == 0:
        raise ValueError()

    # Tokenize
    tokenize_fn = tokenize_fns[0]
    x_tokens = tokenize_fn(x)

    # Compute offsets and lengths
    x_tokens_offs = tokens_offsets(x, x_tokens)
    if None in x_tokens_offs:
        raise ValueError('Tokenizer produced token not found in string')
    if not relative:
        x_tokens_offs = [parent_off + t_off for t_off in x_tokens_offs]
    x_tokens_lens = [len(t) for t in x_tokens]

    if len(tokenize_fns) > 1:
        # Compute recursive offsets for tokens
        x_tokens_offs_recursive = [
            _hierarchical_offsets_recursive(t, tokenize_fns[1:], relative=relative, parent_off=t_off) for t, t_off in
            zip(x_tokens, x_tokens_offs)]
        return tuple(zip(x_tokens_offs, x_tokens_lens, x_tokens_offs_recursive))
    else:
        # Leaf
        return tuple(zip(x_tokens_offs, x_tokens_lens))
