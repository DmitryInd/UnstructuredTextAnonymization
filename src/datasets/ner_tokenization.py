from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizer, PreTrainedTokenizer


class NERTokenizer(ABC):
    def __init__(self, sentence_list: List[List[str]], pad_id: int, pad_flag: bool,
                 max_sent_len: int = None, overlap=32, pretrained_name: str = None):
        """
        :param sentence_list: список предложений для обучения, разбитых на размеченные части
        :param pad_id: id класса 'other'
        :param pad_flag: нужно ли дополнять последовательности токенов до максимальной длины
        :param max_sent_len: максимальная допустимая длина предложений в токенах (+2)
        :param overlap: пересечение последовательных токенизированных отрезков
        :param pretrained_name: путь до сохранённых параметров или название токенизатора
        """
        # Initialisation
        self.pad_id = pad_id
        self.max_sent_len = max_sent_len
        self.overlap = overlap
        self.pad_flag = pad_flag
        # Initialized in the train or load
        self.unknown_token = None
        self.sos_token = None
        self.eos_token = None
        self.pad_token = None

        if pretrained_name is None or Path(pretrained_name).exists():
            self._train(sentence_list, pretrained_name)
        else:
            self._load(pretrained_name)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}
        if self.pad_flag and self.max_sent_len is None:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)

    def __call__(self, sentence_parts: List[str], labels: List[int], force_align: bool = None) -> \
            Tuple[List[int], List[List[int]], List[List[int]]]:
        """
        :param sentence_parts: список частей токенизируемого предложения
        :param labels: список меток, соответствующих этим частям
        :param force_align: форсирующий флаг для (не)приведения предложений к одной длине
        :return: список сдвигов подпоследовательностей в изначальной последовательности,
                 список последовательностей id текстовых токенов и
                 список последовательностей целевых меток, на которые автоматически разбивается входной текст
        """
        token_id_list, label_id_list = [], []
        padding = self.pad_flag if force_align is None else force_align
        for sentence_part, label in zip(sentence_parts, labels):
            id_list = self.simple_encode(sentence_part)
            token_id_list.extend(id_list)
            label_id_list.extend([label] * len(id_list))
        # Segmenting sentence
        offsets, token_segments, label_segments = [], [], []
        if self.max_sent_len is not None:
            i = 0
            while i < len(token_id_list):
                offset, token_id_seg, label_id_seg = self._truncate(token_id_list[i:i + self.max_sent_len + 1],
                                                                    label_id_list[i:i + self.max_sent_len + 1],
                                                                    is_first=i == 0,
                                                                    is_last=i + self.max_sent_len >= len(token_id_list))
                i += offset

                offsets.append(i)
                token_segments.append([self.word2index[self.sos_token]] + token_id_seg +
                                      [self.word2index[self.eos_token]])
                label_segments.append([self.pad_id] + label_id_seg + [self.pad_id])

                i += max(len(token_id_seg) - self.overlap, 1)

            if padding:
                for token_id_seg, label_id_seg in zip(token_segments, label_segments):
                    pad_len = self.max_sent_len + 2 - len(token_id_seg)
                    token_id_seg.extend([self.word2index[self.pad_token]] * pad_len)
                    label_id_seg.extend([self.pad_id] * pad_len)
        else:
            token_segments.append([self.word2index[self.sos_token]] + token_id_list +
                                  [self.word2index[self.eos_token]])
            label_segments.append([self.pad_id] + label_id_list + [self.pad_id])

        return offsets, token_segments, label_segments

    def decode_record(self, offsets: List[int],
                      token_segments_list: List[List[int]],
                      label_segments_list: List[Union[List[int], np.ndarray]]) -> Tuple[List[str], List[int]]:
        """
        Функция совмещает размеченные пересечённые отрывки текста возвращает единую разметку
        :param offsets: последовательность сдвигов по токенам пересекающихся отрезков записи
        :param token_segments_list: последовательность пересекающихся отрезков токенов из одной записи
        :param label_segments_list: последовательность пересекающихся отрезков меток/вероятностей меток токенов из одной записи
        :return: последовательный список частей записи, список соответствующих им меток
        """
        # Firstly, left only tokens between sos and eos ones
        real_length = list(map(lambda x: x.index(self.eos_token), token_segments_list))
        token_segments_list = [token_ids[1:length] for length, token_ids in zip(real_length, token_segments_list)]
        label_segments_list = [labels[1:length] for length, labels in zip(real_length, label_segments_list)]
        # Secondly, lets analyze length of complete text and number of labels
        last_record_id = np.argmax(offsets)
        global_length = offsets[last_record_id] + len(token_segments_list[last_record_id])
        max_label_id = -1
        for labels in label_segments_list:
            if len(labels) == 0:
                continue
            if isinstance(labels, list) and isinstance(labels[0], int):
                max_label_id = max(max_label_id, max(labels))
            elif isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                max_label_id = max(max_label_id, labels[0].shape[1] - 1)

        global_text = [-1] * global_length
        global_labels = np.zeros((global_length, max_label_id + 1), dtype=np.float64)
        global_counter = np.zeros((global_length, max_label_id + 1), dtype=np.int32)
        for offset, token_ids, labels in zip(offsets, token_segments_list, label_segments_list):
            if len(labels) == 0:
                continue
            if isinstance(labels, list) and isinstance(labels[0], int):
                probs = np.eye(max_label_id + 1, dtype=np.float64)[labels]
            elif isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                probs = labels
            else:
                raise ValueError(f"The type of followed labels list is not supported:\n{labels}")

            global_text[offset:offset + len(token_ids)] = token_ids
            global_labels[offset:offset + len(token_ids)] += probs
            global_counter[offset:offset + len(token_ids)] += 1

        return self.decode(global_text, global_labels / global_counter)

    def parse_labels_from_marked_up_log_probs(self, log_probs: torch.Tensor, mark_up: torch.Tensor,
                                              labels_number: Optional[List[int]] = None) -> List[int]:
        """
        Возвращает развёрнутый список предсказанных меток для размеченного тензора логарифмов вероятностей меток
        (разметка определяет границы заложенных последовательностей, для каждой из которых предсказывается одна метка;
         pad_id для не исследуемых позиций)

        log_probs / mark_up: B x L x C
        labels_number: целевое количество последовательностей в каждом примере (для паддинга)
        """
        real_types_list = []
        for i, marks in enumerate(mark_up):
            num = 0
            log_prob = None
            for j, mark in enumerate(marks):
                if mark != self.pad_id:
                    log_prob = log_prob + log_probs[i, j] if log_prob is not None else log_probs[i, j]
                elif mark == self.pad_id and log_prob is not None:
                    real_types_list.append(torch.argmax(log_prob).item())
                    log_prob = None
                    num += 1
                    if num >= labels_number[i]:
                        break
            if labels_number is not None and labels_number[i] > num:
                real_types_list.extend([self.pad_id] * (labels_number[i] - num))
        return real_types_list

    @abstractmethod
    def simple_encode(self, text: str) -> List[int]:
        """
        :param text: строка текста
        :return: представление текста в токенах без вспомогательных токенов
        """
        pass

    @abstractmethod
    def decode(self, token_id_list: List[int], label_list: Union[List[int], np.ndarray]) -> Tuple[List[str], List[int]]:
        """
        :param token_id_list: последовательность id текстовых токенов (для одного предложения)
        :param label_list: последовательность целевых меток/вероятностей меток (для одного предложения)
        :return: последовательный список частей предложения, список соответствующих им меток
        """
        pass

    def get_vocab(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def _train(self, sentence_list: List[List[str]], path_to_pretrained: str = None):
        """
        Обучает внутренний токенизатор или загружает его сохранённый словарь
        (использовать только если обучили токенизатор с помощью этой же функции)
        @param sentence_list: список предложений для обучения, разбитых на размеченные части
        @param path_to_pretrained: путь до сохранённых параметров
        """
        pass

    @abstractmethod
    def _load(self, pretrained_name: str):
        """
        @param pretrained_name: название токенизатора в hugging_face или путь до файла со всеми сохранёнными параметрами
        """
        pass

    def _get_max_length_in_tokens(self, sentence_list: List[List[str]]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, [0], False)[1][0]))
        return max_length

    @abstractmethod
    def _truncate(self, token_id_list: List[int], label_id_list: List[int], is_first=False, is_last=False) \
            -> Tuple[int, List[int], List[int]]:
        """
        Возвращает сдвиг от начала переданной последовательности токенов и её часть,
        обрезанную до максимального размера без дробления слова в начале или в конце.
        Исключение составляет случай, когда первое неполное слово длиннее допустимого пересечения, тогда оно остаётся.

        :param token_id_list: последовательность токенов
        :param label_id_list: последовательность меток токенов
        :param is_first: является ли последовательность первой в разбиваемом тексте
        :param is_last: является ли последовательность заключительной в разбиваемом тексте
        """
        pass


class WordPieceNERTokenizer(NERTokenizer):
    def __init__(self, sentence_list: List[List[str]], pad_id: int, pad_flag: bool, max_sent_len: int = None,
                 overlap=40, pretrained_name: str = None):
        self.hard_max: bool = False
        """Флаг для обработки вероятностей меток, как hard max вероятности"""

        self._tokenizer = None
        super().__init__(sentence_list, pad_id, pad_flag, max_sent_len, overlap, pretrained_name)

    def simple_encode(self, text: str) -> List[int]:
        if isinstance(self._tokenizer, PreTrainedTokenizer):
            return self._tokenizer.encode(text, padding=False, truncation=False)[1:-1]
        elif isinstance(self._tokenizer, Tokenizer):
            return self._tokenizer.encode(text).ids
        else:
            raise ValueError("Incorrect inner tokenizer")

    def decode(self, token_id_list: List[int], label_list: Union[List[int], np.ndarray]) -> Tuple[List[str], List[int]]:
        special_tokens = {self.unknown_token, self.sos_token, self.eos_token, self.pad_token}
        predicted_tokens = []
        predicted_labels = []
        word_tokens = []
        max_label_id = max(label_list) if isinstance(label_list, list) else label_list.shape[1] - 1
        possible_labels = np.zeros((max_label_id + 1, ), dtype=np.float64)
        for token_id, label in zip(token_id_list, label_list):
            token = self.index2word[token_id]
            if word_tokens and token[:2] != "##":
                final_label = np.argmax(possible_labels)
                if predicted_labels and predicted_labels[-1] == final_label:
                    predicted_tokens[-1] += word_tokens
                else:
                    predicted_tokens.append(word_tokens)
                    predicted_labels.append(final_label)
                possible_labels = {}
                word_tokens = []
            if token == self.eos_token:
                break
            if token == self.unknown_token or token not in special_tokens:
                word_tokens += token
                if isinstance(label, int):
                    # If there is only labels, their frequencies are just used
                    probs = np.eye(max_label_id + 1, dtype=np.float64)[label]
                elif isinstance(label, np.ndarray) and len(label.shape) == 1:
                    # If there is soft probabilities of the labels, a beam search on log probabilities is used
                    probs = np.log(label) if not self.hard_max else label
                else:
                    raise ValueError(f'The type of the followed label  is not supported:\n{label}')
                possible_labels += probs
        if word_tokens:
            final_label = np.argmax(possible_labels)
            if predicted_labels and predicted_labels[-1] == final_label:
                predicted_tokens[-1] += word_tokens
            else:
                predicted_labels.append(final_label)
                predicted_tokens.append(word_tokens)
        # Decode token subsequences to substrings, the concatenation of substrings must give a reasonable text
        predicted_tokens = [' ' + self._tokenizer.decode(tokens, skip_special_tokens=True)
                            for tokens in predicted_tokens]
        predicted_tokens[0] = predicted_tokens[0][1:]
        return predicted_tokens, predicted_labels

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def _train(self, sentence_list: List[List[str]], path_to_pretrained: str = None):
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

    def _load(self, pretrained_name: str):
        # Pretrained flag
        self._downloaded = True
        # Download
        self._tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        # Special tokens
        self.unknown_token = self._tokenizer.unk_token
        self.sos_token = self._tokenizer.cls_token
        self.eos_token = self._tokenizer.sep_token
        self.pad_token = self._tokenizer.pad_token

    def _truncate(self, token_id_list: List[int], label_id_list: List[int], is_first=False, is_last=False) \
            -> Tuple[int, List[int], List[int]]:
        offset = 0
        if not is_first:
            for i, token_id in enumerate(token_id_list[:self.overlap]):
                if self.index2word[token_id][:2] != "##":
                    token_id_list = token_id_list[i:]
                    label_id_list = label_id_list[i:]
                    offset = i
                    break

        if not is_last:
            for i, token_id in enumerate(reversed(token_id_list)):
                if self.index2word[token_id][:2] != "##" and len(token_id_list) - i - 1 <= self.max_sent_len + 1:
                    token_id_list = token_id_list[:-i - 1]
                    label_id_list = label_id_list[:-i - 1]
                    break

        return offset, token_id_list, label_id_list
