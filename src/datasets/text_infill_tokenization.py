import json
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Set, Optional

import numpy as np
import regex as re
import torch
from torch import Tensor

from mask.util import convert_masked_docs_to_segments_set


class TargetType(Enum):
    PAD = 0
    CONTEXT = 1
    CONTEXT_SPECIAL = 2
    CONTEXT_INFILL_SEP = 3
    INFILL = 4
    INFILL_SPECIAL = 5


class TextInfillTokenizer(ABC):
    """
    В коде статьи, как я понял, используется оригинальный код bpe токенизатора для GPT2 модели
    вместо реализации из библиотеки transformers. Когда появится свободное время,
    я заменю текущее решение на класс из указанной библиотеки.
    """

    def __init__(self, pretrained_path, mask_types: List[Enum], errors='replace',
                 max_full_ex_len: int = None, max_only_context_len: int = None, overlap: int = None,
                 pad_flag: bool = True):
        """
        :param pretrained_path: директория с сохранёнными параметрами токенизаторов
        :param mask_types: типы масок, содержащихся во входных данных
        :param errors: способ обработки ошибок
        :param max_full_ex_len: максимальное количество токенов в примере
        :param max_only_context_len: максимальное количество токенов для запроса без ответа
        :param overlap: пересечение последовательных частей предложений
        :param pad_flag: дополнять ли все примеры паддингом до max_full_ex_len
        """
        self._load(pretrained_path, errors)
        # Add to tokenizer vocabulary technical characters
        self.pad_id = None
        self.start_infill_id = None
        self.end_infill_id = None
        self.mask_type_to_id = None
        self._add_special_characters(mask_types)
        # Parameters for splitting text
        self.max_full_ex_len = max_full_ex_len  # 1024
        self.max_only_context_len = max_only_context_len  # 3/4 * max_full_ex_len
        self.overlap = max_full_ex_len * 3 // 16 if overlap is None and max_full_ex_len is not None else overlap
        self.pad_flag = pad_flag

    def __call__(self, text: str, masks: List[List[Tuple[Enum, int, int]]], with_answers=True) \
            -> Tuple[List[np.array], List[np.array]]:
        """
        :param text: текст в формате строки
        :param masks: список наборов масок для текста в формате (тип, сдвиг, длина)
        :return: список массивов токенов запросов для модели, список разметки запросов для CrossEntropy
        """
        context_and_answers = self._get_context_and_answers(text, masks)
        inputs, tts = self._build_queries(context_and_answers, with_answers)  # tts = target types
        return inputs, tts

    def align_inputs(self, token_inputs: List[torch.Tensor], markups: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Дополняет входные последовательности паддингом до максимальной длины среди них, после чего склеивает в один массив.
        :param token_inputs: список массивов токенов запросов для модели
        :param markups: список разметки запросов для CrossEntropy
        :return: массив токенов запросов для модели, массив разметки запросов для CrossEntropy
        """
        full_len = max(map(len, token_inputs))
        new_token_inputs = torch.full((len(token_inputs), full_len), self.pad_id, dtype=token_inputs[0].dtype)
        new_markups = torch.full((len(token_inputs), full_len), TargetType.PAD.value, dtype=markups[0].dtype)
        for i, token_input, markup in enumerate(zip(token_inputs, markups)):
            new_token_inputs[i, :len(token_input)] = token_input
            new_markups[i, :len(markup)] = markup

        return new_token_inputs, new_markups

    def parse_answers(self, prediction: Tensor, target: Optional[Tensor] = None) -> List[str]:
        """
        :param prediction: tensor[int] 1 x len - выход модели для одного примера
        :param target: tensor[int] 1 x len - целевой ответ для одного примера
                       (опционально, чтобы выравнять количество ответов в предсказанном и целевом результате для CER)
        :return: список декодированных слов, предсказанных моделью для заполнения пропусков
        """
        if target is None:
            start = prediction.tolist().index(self.start_infill_id)
        else:
            start = target.tolist().index(self.start_infill_id)

        answers = prediction[start + 1:].tolist()
        answers_list = []
        while answers:
            try:
                end_answer = answers.index(self.end_infill_id)
            except ValueError:
                end_answer = len(answers)
            answers_list.append(self.decode(answers[:end_answer]))
            answers = answers[end_answer + 1:]

        if target is not None:
            mask_n = (target == self.end_infill_id).sum()
            answers_list = answers_list[:mask_n]
            answers_list.extend(["" for _ in range(mask_n)])

        return answers_list

    @abstractmethod
    def encode(self, text) -> List[int]:
        """
        Кодирует текст без разделения его на подпоследовательности
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def _load(self, pretrained_path, errors):
        """
        @param pretrained_path: директория с сохранёнными параметрами токенизаторов
        @param errors: способ обработки ошибок
        """
        pass

    @abstractmethod
    def _add_special_characters(self, mask_types: List[Enum]) -> int:
        """
        :param mask_types: список типов масок в формате Enum
        :return: новое количество токенов в токенизаторе
        """
        pass

    def _get_context_and_answers(self, doc: str, char_masks: List[List[Tuple[Enum, int, int]]]) \
            -> List[Tuple[list, List[Tuple[Enum, list]]]]:
        """
        :param doc: текст в формате строки
        :param char_masks: список наборов масок для текста в формате [(тип, сдвиг, длина), ...]
        :return: [(список токенов документа без замаскированных символов, список масок в формате (тип, список токенов)), ...]
        """
        _, category_lists, segments_lists = convert_masked_docs_to_segments_set([('', doc, char_masks)], other_label='')

        contexts_and_answers = []
        for i, (category_list, segments_list) in enumerate(zip(category_lists, segments_lists)):
            context = []
            answers = []
            j = 0
            for category, segment in zip(category_list, segments_list):
                segment = self.encode(segment)  # Convert text to tokens
                if category:
                    # Such an over complication has occurred historically, but now we can verify the results
                    j += 1
                    mask_type = char_masks[i][j][0]
                    assert category == mask_type.name, "The mask category does not match the mask type"

                    context.append(self.mask_type_to_id(mask_type))
                    answers.append((mask_type, segment))
                else:
                    context += segment
            contexts_and_answers.append((context, answers))

        return contexts_and_answers

    def _build_queries(self, contexts_and_answers: List[Tuple[list, List[Tuple[Enum, list]]]], with_answers=True) \
            -> Tuple[List[np.array], List[np.array]]:
        """
        :param contexts_and_answers: [(список токенов документа без замаскированных символов, список масок в формате (тип, список токенов)), ...]
        :param with_answers: добавлять ли в запросы ответ модели с замаскированными словами
        :return: список массивов токенов запросов для модели, список разметки запросов для CrossEntropy
        """
        special_ids = set([self.start_infill_id, self.end_infill_id] + list(self.mask_type_to_id.values()))
        pad_flag = self.max_full_ex_len is not None and self.pad_flag
        overlap = 0 if self.overlap is None else self.overlap
        inputs = []
        markups = []  # Markups for cross-entropy loss
        for context, answers in contexts_and_answers:
            # Find first mask occurrence
            while answers:
                context_len, mask_num, last_mask_pos = self._find_the_longest_query(context, answers, with_answers)
                if mask_num == 0:
                    context = context[max(context_len - overlap, 1):]
                    continue

                # Create example
                # (Masked) Context
                # Example: She ate <?> for <?>
                example = context[: context_len]
                # Context / answer separator
                # Example: <S>
                example += [self.start_infill_id]
                if with_answers:
                    # Answers
                    # Example: cereal<E>breakfast<E>
                    for mask_type, answer in answers[:mask_num]:
                        example += answer
                        example += [self.end_infill_id]

                assert len(example) <= self.max_full_ex_len

                full_len = self.max_full_ex_len if pad_flag else len(example)
                token_input = np.full((full_len,), self.pad_id)
                markup = np.full((full_len,), TargetType.PAD.value)

                # Find special tokens
                context_special_idxs = [l for l, t in enumerate(example) if l < context_len and t in special_ids]
                infill_special_idxs = [l for l, t in enumerate(example) if l > context_len and t in special_ids]

                # Store example in output array
                token_input[:len(example)] = example
                inputs.append(token_input)

                # Store target types in output array
                markup[:context_len] = TargetType.CONTEXT.value
                for l in context_special_idxs:
                    markup[l] = TargetType.CONTEXT_SPECIAL.value
                markup[context_len:context_len + 1] = TargetType.CONTEXT_INFILL_SEP.value
                markup[context_len + 1: len(example)] = TargetType.INFILL.value
                for l in infill_special_idxs:
                    markup[l] = TargetType.INFILL_SPECIAL.value
                markups.append(markup)

                # Shift context
                context = context[max(last_mask_pos + 1, context_len - overlap):]
                answers = answers[mask_num:]

        return inputs, markups

    def _find_the_longest_query(self, context: List[int], answers: List[Tuple[Enum, List[int]]], with_answer=True):
        """
        :param context: список токенов текста (с токенами масок)
        :param answers: список пропущенных последовательностей в формате (тип, список токенов)
        :param with_answer: нужно ли добавлять в запрос ответ
        :return: максимально допустимая длина контекста, количество масок в ней, позиция последней маски в контексте
        """
        if self.max_only_context_len is not None and not with_answer:
            max_sent_len = self.max_only_context_len
        elif self.max_full_ex_len is not None:
            max_sent_len = self.max_full_ex_len
        else:
            max_sent_len = 0

        i, j = 0, 0  # context cursor, answers cursor
        last_mask_pos = -1
        query_len = 1
        for token in context:
            if max_sent_len and query_len >= max_sent_len:
                break
            query_len += 1
            if token in self.mask_type_to_id.values():
                if with_answer:
                    answer = answers[j][1]
                else:
                    answer = []
                # (context + <S> + answers) + answer + <E>
                new_query_len = query_len + len(answer)
                if with_answer:
                    new_query_len += 1
                if max_sent_len and new_query_len > max_sent_len:
                    break
                query_len = new_query_len
                last_mask_pos = i
                j += 1
            i += 1

        return i, j, last_mask_pos


class OfficialGPT2TextInfillTokenizer(TextInfillTokenizer):
    """
    В коде статьи, как я понял, используется оригинальный код bpe токенизатора для GPT2 модели
    вместо реализации из библиотеки transformers. Когда появится свободное время,
    я заменю текущее решение на класс из указанной библиотеки.
    """
    def encode(self, text) -> List[int]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self._bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return len(self.encoder)

    def _load(self, pretrained_path, errors):
        # Load pretrained tokenizer for GPT2
        with open(Path(pretrained_path) / Path('encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(Path(pretrained_path) / Path('vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        # Tokenizer main entities
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        # Pattern for word splitting
        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @staticmethod
    @lru_cache()
    def _bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a significant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
            range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _add_special_characters(self, mask_types: List[Enum]) -> int:
        self.pad_id = 0
        base_vocab_size = len(self.encoder)
        self.start_infill_id = base_vocab_size + 0
        self.end_infill_id = base_vocab_size + 1
        additional_ids_to_tokens = {
            self.start_infill_id: '<|startofinfill|>',
            self.end_infill_id: '<|endofinfill|>'
        }
        self.mask_type_to_id = {}
        for i, t in enumerate(mask_types):
            t_id = base_vocab_size + 2 + i
            t_tok = '<|infill_{}|>'.format(t.name.lower())
            additional_ids_to_tokens[t_id] = t_tok
            self.mask_type_to_id[t] = t_id

        additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}
        self.encoder.update(additional_tokens_to_ids)
        self.decoder.update(additional_ids_to_tokens)
        vocab_size_after = len(self.encoder)

        return vocab_size_after

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
