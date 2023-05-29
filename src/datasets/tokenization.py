from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizer, PreTrainedTokenizer


class WordPieceTokenizer:
    def __init__(self, sentence_list: List[List[str]], other_id: int, pad_flag: bool,
                 max_sent_len: int = None, pretrained_name: str = None):
        """
        :param sentence_list: список предложений для обучения, разбитых на размеченные части
        :param other_id: id класса 'other'
        :param pad_flag: нужно ли приводить последовательности токенов к одной длине
        :param max_sent_len: максимальная допустимая длина предложений в токенах (+2)
        :param pretrained_name: путь до сохранённых параметров или название токенизатора
        """
        # Initialisation
        self.other_id = other_id
        self.max_sent_len = max_sent_len
        self.pad_flag = pad_flag
        if pretrained_name is None or Path(pretrained_name).exists():
            self._tokenizer = self._train(sentence_list, pretrained_name)
        else:
            self._tokenizer = self._load(pretrained_name)
        if self.pad_flag and self.max_sent_len is None:
            self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
        # Preparing dictionaries mapping tokens and ids
        self.word2index = self._tokenizer.get_vocab()
        self.index2word = {w_id: word for word, w_id in self.word2index.items()}

    def __call__(self, sentence_parts: List[str], labels: List[int], force_padding: bool = None):
        """
        :param sentence_parts: список частей токенизируемого предложения
        :param labels: список меток, соответствующих этим частям
        :param force_padding: форсирующий флаг о приведении предложений к одной длине
        :return: последовательность id текстовых токенов и последовательность целевых меток (для одного предложения)
        """
        token_id_list = [self.word2index[self.sos_token]]
        label_id_list = [self.other_id]
        padding = self.pad_flag if force_padding is None else force_padding
        for sentence_part, label in zip(sentence_parts, labels):
            if self._downloaded:
                id_list = self._tokenizer.encode(sentence_part, padding=False, truncation=False)[1:-1]
            else:
                id_list = self._tokenizer.encode(sentence_parts).ids
            token_id_list.extend(id_list)
            label_id_list.extend([label] * len(id_list))
            if padding and (self.max_sent_len is not None) and (len(token_id_list) > self.max_sent_len + 1):
                token_id_list, label_id_list = self._truncate(token_id_list[:self.max_sent_len + 2],
                                                              label_id_list[:self.max_sent_len + 2])
                break

        token_id_list.append(self.word2index[self.eos_token])
        label_id_list.append(self.other_id)
        if padding and (self.max_sent_len is not None) and (len(token_id_list) < self.max_sent_len + 2):
            pad_len = self.max_sent_len + 2 - len(token_id_list)
            token_id_list.extend([self.word2index[self.pad_token]] * pad_len)
            label_id_list.extend([self.other_id] * pad_len)
        return token_id_list, label_id_list

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
        Возвращает последовательность токенов, обрезанную до максимального размера без дробления слова в конце
        """
        for i, token_id in enumerate(reversed(token_id_list)):
            if self.index2word[token_id][:2] != "##" and len(token_id_list) - i - 1 <= self.max_sent_len + 1:
                token_id_list = token_id_list[:-i - 1]
                label_id_list = label_id_list[:-i - 1]
                break

        return token_id_list, label_id_list

    def _get_max_length_in_tokens(self, sentence_list: List[List[str]]) -> int:
        max_length = 0
        for sentence in sentence_list:
            max_length = max(max_length, len(self(sentence, [0], False)))
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
