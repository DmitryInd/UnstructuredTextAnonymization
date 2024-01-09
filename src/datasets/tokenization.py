import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import regex as re
from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizer, PreTrainedTokenizer

from mask.util import align_char_mask_to_tokens, apply_masked_spans


class WordPieceTokenizer:
    def __init__(self, sentence_list: List[List[str]], pad_id: int, pad_flag: bool,
                 max_sent_len: int = None, overlap=40, pretrained_name: str = None):
        """
        :param sentence_list: список предложений для обучения, разбитых на размеченные части
        :param pad_id: id класса 'other'
        :param pad_flag: нужно ли дополнять последовательности токенов до максимальной длины
        :param max_sent_len: максимальная допустимая длина предложений в токенах (+2)
        :param overlap: пересечение последовательных частей предложений
        :param pretrained_name: путь до сохранённых параметров или название токенизатора
        """
        # Initialisation
        self.pad_id = pad_id
        self.max_sent_len = max_sent_len
        self.overlap = overlap
        self.pad_flag = pad_flag
        if pretrained_name is None or Path(pretrained_name).exists():
            self._tokenizer = self._train(sentence_list, pretrained_name)
        else:
            self._tokenizer = self._load(pretrained_name)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}
        if self.pad_flag and self.max_sent_len is None:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)

    def __call__(self, sentence_parts: List[str], labels: List[int], force_align: bool = None) -> \
            Tuple[List[List[int]], List[List[int]]]:
        """
        :param sentence_parts: список частей токенизируемого предложения
        :param labels: список меток, соответствующих этим частям
        :param force_align: форсирующий флаг о приведении предложений к одной длине
        :return: список последовательностей id текстовых токенов и
                 список последовательностей целевых меток, на которые автоматически разбивается входной текст
        """
        token_id_list, label_id_list = [], []
        padding = self.pad_flag if force_align is None else force_align
        for sentence_part, label in zip(sentence_parts, labels):
            if self._downloaded:
                id_list = self._tokenizer.encode(sentence_part, padding=False, truncation=False)[1:-1]
            else:
                id_list = self._tokenizer.encode(sentence_parts).ids
            token_id_list.extend(id_list)
            label_id_list.extend([label] * len(id_list))
        # Segmenting sentence
        token_segments, label_segments = [], []
        if (self.max_sent_len is not None) and force_align:
            for i in range(0, len(token_id_list), self.max_sent_len - self.overlap):
                token_id_seg, label_id_seg = self._truncate(token_id_list[i:i + self.max_sent_len + 1],
                                                            label_id_list[i:i + self.max_sent_len + 1])
                token_segments.append([self.word2index[self.sos_token]] + token_id_seg +
                                      [self.word2index[self.eos_token]])
                label_segments.append([self.pad_id] + label_id_seg + [self.pad_id])
        else:
            token_segments.append([self.word2index[self.sos_token]] + token_id_list +
                                  [self.word2index[self.eos_token]])
            label_segments.append([self.pad_id] + label_id_list + [self.pad_id])

        if padding and (self.max_sent_len is not None):
            for token_id_seg, label_id_seg in zip(token_segments, label_segments):
                pad_len = self.max_sent_len + 2 - len(token_id_seg)
                token_id_seg.extend([self.word2index[self.pad_token]] * pad_len)
                label_id_seg.extend([self.pad_id] * pad_len)
        return token_segments, label_segments

    def decode(self, token_id_list: List[int], label_list: List[int]):
        """
        :param token_id_list: последовательность id текстовых токенов (для одного предложения)
        :param label_list: последовательность целевых меток (для одного предложения)
        :return: последовательный список слов из предложения, список соответствующих им меток
        """
        special_tokens = {self.unknown_token, self.sos_token, self.eos_token, self.pad_token}
        predicted_tokens = []  # self._tokenizer.decode(token_id_list, skip_special_tokens=True).split()
        predicted_labels = []
        word = ""
        possible_labels = {}
        for token_id, label_id in zip(token_id_list, label_list):
            token = self.index2word[token_id]
            if word != "" and (len(token) < 2 or token[:2] != "##"):
                final_label = sorted([(num, tag) for tag, num in possible_labels.items()])[-1][1]
                predicted_labels.append(final_label)
                predicted_tokens.append(word)
                possible_labels = {}
                word = ""
            if token == self.eos_token:
                break
            if token == self.unknown_token or token not in special_tokens:
                word += token[2:] if token[:2] == "##" else token
                possible_labels[label_id] = possible_labels.get(label_id, 0) + 1
        if word != "":
            final_label = sorted([(num, tag) for tag, num in possible_labels.items()])[-1][1]
            predicted_labels.append(final_label)
            predicted_tokens.append(word)
        return predicted_tokens, predicted_labels

    def _truncate(self, token_id_list: List[int], label_id_list: List[int]):
        """
        Возвращает последовательность токенов, обрезанную до максимального размера
        без дробления слова в начале или в конце
        """
        for i, token_id in enumerate(token_id_list):
            if self.index2word[token_id][:2] != "##":
                token_id_list = token_id_list[i:]
                label_id_list = label_id_list[i:]
                break

        for i, token_id in enumerate(reversed(token_id_list)):
            if self.index2word[token_id][:2] != "##" and len(token_id_list) - i - 1 <= self.max_sent_len + 1:
                token_id_list = token_id_list[:-i - 1]
                label_id_list = label_id_list[:-i - 1]
                break

        return token_id_list, label_id_list

    def _get_max_length_in_tokens(self, sentence_list: List[List[str]]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, [0], False)[0][0]))
        return max_length

    def _train(self, sentence_list: List[List[str]], path_to_pretrained: str = None) -> Tokenizer:
        # Pretrained flag
        self._downloaded = False
        # Special tokens
        self.unknown_token = "[UNK]"
        self.sos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.pad_token = "[PAD]"
        # Initialization
        if path_to_pretrained is None:
            self._tokenizer = Tokenizer(WordPiece(unk_token=self.unknown_token))
        else:
            self._tokenizer = Tokenizer(WordPiece.from_file(path_to_pretrained))
        self._tokenizer.pre_tokenizer = Whitespace()
        self._tokenizer.decoder = decoders.WordPiece()
        # Training
        if path_to_pretrained is None:
            trainer = WordPieceTrainer(
                special_tokens=[self.unknown_token, self.sos_token, self.eos_token, self.pad_token])
            self._tokenizer.train_from_iterator(sentence_list, trainer)
        return self._tokenizer

    def _load(self, pretrained_name: str) -> PreTrainedTokenizer:
        # Pretrained flag
        self._downloaded = True
        # Download
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        # Special tokens
        self.unknown_token = self._tokenizer.unk_token
        self.sos_token = self._tokenizer.cls_token
        self.eos_token = self._tokenizer.sep_token
        self.pad_token = self._tokenizer.pad_token
        return self._tokenizer


class TargetType(Enum):
    PAD = 0
    CONTEXT = 1
    CONTEXT_SPECIAL = 2
    CONTEXT_INFILL_SEP = 3
    INFILL = 4
    INFILL_SPECIAL = 5
    INFILL_REDUNDANT = 6


class OfficialGPT2Tokenizer:
    """
    В коде статьи, как я понял, используется оригинальный код bpe токенизатора для GPT2 модели
    вместо реализации из библиотеки transformers. Когда появится свободное время,
    я заменю текущее решение на класс из указанной библиотеки.
    """

    def __init__(self, tokenizer_dir, mask_types: List[Enum], errors='replace',
                 max_sent_len: int = None, overlap: int = None):
        """
        :param tokenizer_dir: директория с сохранёнными параметрами токенизаторов
        :param mask_types: типы масок, содержащихся во входных данных
        :param errors: способ обработки ошибок
        """
        # Load pretrained tokenizer for GPT2
        with open(Path(tokenizer_dir) / Path('official_gpt2_encoder/encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(Path(tokenizer_dir) / Path('official_gpt2_encoder/vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        # Tokenizer main entities
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        # Pattern for word splitting
        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # Add to tokenizer vocabulary technical characters
        self.start_infill_id = None
        self.end_infill_id = None
        self.mask_type_to_id = None
        self.add_special_characters(mask_types)
        # Parameters for splitting text
        self.max_sent_len = max_sent_len  # 768
        self.overlap = overlap
        if max_sent_len is not None:
            self.overlap = max_sent_len // 4 if overlap is None else overlap

    @staticmethod
    @lru_cache()
    def bytes_to_unicode():
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

    def add_special_characters(self, mask_types: List[Enum]) -> int:
        """
        :param mask_types: список типов масок в формате Enum
        :return: новое количество токенов в токенизаторе
        """
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

    def __call__(self, text: str, masks: List[List[Tuple[Enum, int, int]]]) -> List[List[int]]:
        """
        :param text: текст в формате строки
        :param masks: список наборов масок для текста в формате (тип, сдвиг, длина)
        :return: список последовательностей токенов и соответствующие им маски
        """
        tokens_ids = self.encode(text)
        train_inputs, train_tts, train_num_docs = self.doc_and_char_masks_to_input_and_tts(text, tokens_ids, masks)
        token_mask_sets = []
        for mask in masks:
            pass
        # TODO Добавить разделение на последовательности максимальной длины с пересечением
        if max_num_examples is not None:
            set_random_seed(args.seed)
            example_ids = random.sample(list(range(inputs.shape[0])), max_num_examples)
            inputs = np.take(inputs, example_ids, axis=0)
            tts = np.take(tts, example_ids, axis=0)
        return [tokens_ids]

    def encode(self, text) -> List[int]:
        """
        Кодирует текст без разделения его на подпоследовательности
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    @staticmethod
    def get_pairs(word: Tuple[str]) -> Set[Tuple[str, str]]:
        """Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)

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
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def decode(self, tokens: List[int]) -> str:
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def doc_and_char_masks_to_input_and_tts(self, doc, doc_tokens_ids, char_masks, skip_naive_incomplete=False):
        # Get text tokens from token_ids
        raw_tokens = [self.decoder[token_id] for token_id in doc_tokens_ids]
        doc_tokens = [bytearray([self.byte_decoder[c] for c in token]).decode('utf-8', errors=self.errors)
                      for token in raw_tokens]

        # Align character masks to tokens
        tok_masks = []
        for char_mask in char_masks:
            try:
                tok_mask = align_char_mask_to_tokens(doc, doc_tokens, char_mask)
            except:
                # error_to_count['Failed to align character-level mask to tokens'] += 1
                continue
            tok_masks.append(tok_mask)

        # Apply masks
        contexts_and_answers = []
        for tok_mask in tok_masks:
            try:
                ca = apply_masked_spans(doc_tokens_ids, tok_mask, self.mask_type_to_id)
            except:
                # error_to_count['Failed to apply mask'] += 1
                continue
            contexts_and_answers.append((tok_mask, ca))

        special_ids = set([self.start_infill_id, self.end_infill_id] + list(self.mask_type_to_id.values()))
        # TODO выделить в отдельную функцию
        inputs = np.zeros((len(contexts_and_answers), sequence_length), dtype=np.uint16)
        tts = np.full((len(contexts_and_answers), sequence_length), TargetType.PAD.value, dtype=np.uint8)
        for i, (mask, (context, answers)) in enumerate(contexts_and_answers):
            # Create example
            example = []
            # (Masked) Context
            # Example: She ate <?> for <?>
            example += context
            # Context / answer separator
            context_len = len(example)
            # Example: <S>
            example += [self.start_infill_id]
            # Answers
            # Example: cereal<E>breakfast<E>
            for mask_type, answer in answers:
                example += answer
                example += [self.end_infill_id]

            if len(example) > sequence_length:
                example = example[:sequence_length]
                # warning_to_count['Example longer than sequence length'] += 1

            # Find special tokens
            context_special_idxs = [l for l, t in enumerate(example) if l < context_len and t in special_ids]
            infill_special_idxs = [l for l, t in enumerate(example) if l > context_len and t in special_ids]

            # Store example in output array
            if len(example) > 0 and (
                    min(example) < np.iinfo(inputs.dtype).min or max(example) > np.iinfo(inputs.dtype).max):
                raise ValueError('Example cannot be stored in numpy array')
            inputs[i, :len(example)] = example

            # Store target types in output array
            tts[i, :context_len] = TargetType.CONTEXT.value
            for l in context_special_idxs:
                tts[i, l] = TargetType.CONTEXT_SPECIAL.value
            tts[i, context_len:context_len + 1] = TargetType.CONTEXT_INFILL_SEP.value
            tts[i, context_len + 1:len(example)] = TargetType.INFILL.value
            for l in infill_special_idxs:
                tts[i, l] = TargetType.INFILL_SPECIAL.value

        return inputs, tts
