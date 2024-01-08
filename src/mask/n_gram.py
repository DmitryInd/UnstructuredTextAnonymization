import random
from enum import Enum
from functools import lru_cache
from typing import List, Tuple

from nltk import sent_tokenize, word_tokenize as nltk_word_tokenize

from base import MaskFn
from util import tokens_offsets


class MaskNgramType(Enum):
    NGRAM = 0


class NgramsMaskFn(MaskFn):
    def __init__(self, p=0.03, max_ngram_mask_length=8):
        try:
            sent_tokenize('Ensure punkt installed.')
        except:
            raise ValueError('Need to call nltk.download(\'punkt\')')
        self.p = p
        self.max_ngram_mask_length=max_ngram_mask_length

    @staticmethod
    def mask_types() -> List[str]:
        return list(MaskNgramType)

    def mask_type_serialize(self, m_type):
        return m_type.name.lower()

    def mask(self, doc, mask_p=None, max_ngram_mask_length=8) -> List[Tuple[MaskNgramType, int, int]]:
        """
        Маскирует случайные n-граммы слов в документе. Длина n выбирается случайным образом равномерно между 1 и
        указанной максимальной длиной.
        :param doc: текст документа в формате строки
        :param mask_p: вероятность замаскировать n-грамму, начиная с указанного слова
        :param max_ngram_mask_length: максимальная длина замаскированной n-граммы
        :return: список троек:
        (тип замаскированного объекта, сдвиг на начало замаскированного объекта, длина замаскированного объекта)
        """
        doc_offs = doc_to_hierarchical_offsets(doc)

        mask_p = self.p if mask_p is None else mask_p
        max_ngram_mask_length = self.max_ngram_mask_length if max_ngram_mask_length is None else max_ngram_mask_length

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
                    if _trial(mask_p):
                        # Mask ngram starting at word
                        max_k = min(len(w_offs) - w_i, max_ngram_mask_length)
                        assert max_k > 0
                        k = random.randint(1, max_k)
                        last_w_off, last_w_len = w_offs[w_i + k - 1]
                        masked_spans.append(
                            (MaskNgramType.NGRAM, w_off, last_w_off + last_w_len - w_off))
                        w_i += k
                    else:
                        w_i += 1

        return masked_spans


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
