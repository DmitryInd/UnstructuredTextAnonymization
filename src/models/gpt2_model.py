from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU
from torch.nn.functional import log_softmax
from torchmetrics import Recall
from torchmetrics.text import CharErrorRate
from transformers import GPT2LMHeadModel

from datasets.ner_tokenization import NERTokenizer
from datasets.text_infill_tokenization import TextInfillTokenizer, TargetType
from models.bert_model import PretrainedBertNER


class PretrainedGPT2TextInfilling(pl.LightningModule):
    def __init__(self, pretrained_name: str, vocab_size: int, train_context: float,
                 lr: float, total_steps: int, adaptation_part: int, div_factor: int,
                 end_infill_id=None, target_types_pred=0.0, real_types_match=0.0, num_classes=1, **kwargs):
        """
        :param pretrained_name: название предобученной GPT2 модели из hugging face hub
        :param vocab_size: итоговый размер словаря (с добавлением или удалением части токенов)
        :param train_context: вес функции потерь на контексте (0. - отключение)
        :param lr: максимальный learning rate
        :param total_steps: полное количество шагов обучения: ~ кол-во эпох * размер батча
        :param adaptation_part: доля эпох для обновления весов с низким learning rate
        :param div_factor: максимальный делитель, на который уменьшается learning rate в OneCycle подходе
        :param end_infill_id: id end_infill токена для инференса модели
        :param target_types_pred: вес функции потерь для предсказания целевых меток для сгенерированных данных
        :param real_types_match: вес функции потерь для предсказания истинных меток для сгенерированных данных
        :param num_classes: количество типов замаскированных данных, генерируемых моделью (num_classes > max_type_id)
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self.type_head = Sequential(
            Linear(self.model.lm_head.in_features, self.model.lm_head.in_features // 2),
            LeakyReLU(),
            Linear(self.model.lm_head.in_features // 2, num_classes),
        )
        self.end_infill_id = end_infill_id
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        self.adaptation_part = adaptation_part
        self.train_context = train_context
        self.target_types_pred = target_types_pred
        self.real_types_match = real_types_match
        # Additional modules for training
        self.tokenizer: Optional[TextInfillTokenizer] = None
        self.ner_model: Optional[PretrainedBertNER] = None
        self.ner_tokenizer: Optional[NERTokenizer] = None
        """Токенизатор для NER модели со следующими параметрами: pad_id=-1, overlap=0"""
        # Metrics for quality evaluation
        self.train_cer = CharErrorRate()
        self.val_cer = CharErrorRate()
        self.train_target_type_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_target_type_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.train_real_type_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_real_type_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=-1)

    def forward(self, x, gen_types=False, **kwargs):
        gen_types = gen_types and self.type_head is not None
        x = self.model(x, output_hidden_states=gen_types, **kwargs)  # B, L, C
        types_pred = None
        if gen_types:
            types_pred = self.type_head(x.hidden_states[-1])  # B, L, C
        return x.logits, types_pred

    def configure_optimizers(self):
        # Similar to the article
        params = list(self.named_parameters())
        no_decay = ['bias', 'ln']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-2
            },
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.lr,
            pct_start=self.adaptation_part,
            anneal_strategy='cos',
            final_div_factor=self.div_factor
        )
        return [optimizer], [scheduler]

    def classic_step(self, inputs, tts, target_type_recall, real_type_recall, cer):
        target_context = self.tts_to_targets(inputs, tts, [TargetType.CONTEXT])
        target_infill = self.tts_to_targets(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])
        logits, types_logits = self.forward(inputs[:, :-1], gen_types=bool(self.target_types_pred or
                                                                           self.real_types_match))
        logits = logits.transpose(2, 1)  # B, L, C -> B, C, L
        types_logits = types_logits.transpose(2, 1)  # B, L, C -> B, C, L
        # Get hard predictions for answer and their types
        answers_starts = torch.nonzero(inputs == self.tokenizer.start_infill_id).tolist()
        hard_pred = self.get_hard_prediction(inputs, logits, answers_starts)
        hard_types = torch.argmax(types_logits, dim=-2)
        # Compute main generation loss
        loss_infill = self.criterion(logits, target_infill[:, 1:])
        loss = loss_infill
        if self.train_context:
            loss_context = self.criterion(logits, target_context[:, 1:])
            loss += self.train_context * loss_context
        # Compute target labels loss
        if self.target_types_pred and self.tokenizer is not None:
            target_types_list = inputs[tts == TargetType.CONTEXT_SPECIAL.value].cpu()
            target_types_list = np.vectorize(lambda x: self.tokenizer.id_to_mask_type[x].value)(target_types_list).tolist()
            target_types = self.tokenizer.mark_up_types(inputs, target_types_list)[:, :-1]
            loss += self.target_types_pred * self.criterion(types_logits, target_types)
            target_type_recall.update(hard_types, target_types)
        # Compute real labels loss
        if (self.real_types_match and self.tokenizer is not None
                and self.ner_model is not None and self.ner_tokenizer is not None):
            assert self.ner_tokenizer.pad_id == -1, "Pad id for NER tokenizer must be -1"
            assert self.ner_tokenizer.overlap == 0, "Overlap of NER tokenizer must be 0"
            sample_answers = [self.tokenizer.parse_answers(gen, st[1]) for st, gen in zip(answers_starts, hard_pred)]
            num_answers = [len(answers) for answers in sample_answers]
            pseudo_labels = [[1] * num for num in num_answers]
            ner_input, ner_labels = self._prepare_input_for_ner(sample_answers, pseudo_labels)
            with torch.no_grad():
                log_probs = log_softmax(self.ner_model(ner_input), dim=-1)
            real_types_list = self.ner_tokenizer.parse_labels_from_marked_up_log_probs(log_probs, ner_labels,
                                                                                       num_answers)
            real_types = self.tokenizer.mark_up_types(hard_pred, real_types_list)[:, :-1]
            loss += self.real_types_match * self.criterion(types_logits, real_types)
            real_type_recall.update(hard_types, real_types)
        # Compute cer statistics
        if self.tokenizer is not None:
            answers_numbers = (inputs == self.tokenizer.end_infill_id).sum(dim=-1).tolist()
            for answers_start, answers_number, orig, gen in zip(answers_starts, answers_numbers, inputs, hard_pred):
                cer.update(self.tokenizer.parse_answers(gen, answers_start[1], answers_number),
                           self.tokenizer.parse_answers(orig, answers_start[1]))
        return loss

    def training_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        loss = self.classic_step(inputs, tts,
                                 self.train_target_type_recall, self.train_real_type_recall, self.train_cer)

        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_recall', self.train_target_type_recall, on_step=False, on_epoch=True, logger=True,
                 prog_bar=True)
        self.log('train_recall', self.train_real_type_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_cer', self.train_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        loss = self.classic_step(inputs, tts,
                                 self.val_target_type_recall, self.val_real_type_recall, self.val_cer)

        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_recall', self.val_target_type_recall, on_step=False, on_epoch=True, logger=True,
                 prog_bar=True)
        self.log('val_recall', self.val_real_type_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_cer', self.val_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)

    @torch.no_grad()
    def inference(self, inputs: torch.Tensor, tts: torch.Tensor):
        masks_number = (tts == TargetType.CONTEXT_SPECIAL.value).sum(dim=1)
        positions = (tts != TargetType.PAD.value).sum(dim=1)
        finished = set()
        while len(finished) < inputs.shape[0]:
            logits, _ = self.forward(inputs)  # B, L, C
            hard_pred = torch.argmax(logits, dim=-1)
            for i, row in enumerate(hard_pred):
                pos = positions[i]
                if pos >= inputs.shape[1] or masks_number[i] <= 0:
                    finished.add(i)
                    continue
                inputs[i, pos] = row[pos - 1]
                if row[pos - 1] == self.end_infill_id:
                    masks_number[i] -= 1
                positions[i] += 1

        return inputs

    @staticmethod
    def tts_to_targets(inputs, tts, label_tts):
        """
        Заменяет нецелевые токены в inputs на -1
        """
        selector = torch.zeros_like(inputs, dtype=torch.bool)
        for tt in label_tts:
            selector |= (tts == tt.value)
        return torch.where(
            selector,
            inputs,
            torch.full_like(inputs, -1))

    def get_hard_prediction(self, inputs: torch.Tensor, logits: torch.Tensor, answers_starts: List[int] = None):
        assert len(inputs.shape) == 2 and len(logits.shape) == 3, "Shapes of inputs and logits must be BxL and BxCxL"
        assert inputs.shape[0] == logits.shape[0] and inputs.shape[1] - 1 == logits.shape[2], \
            "Shapes of inputs and logits do not match"
        if answers_starts is None:
            answers_starts = torch.nonzero(inputs == self.tokenizer.start_infill_id).tolist()

        hard_pred = torch.argmax(logits, dim=-2)
        for i, start in answers_starts:
            hard_pred[i, :start] = inputs[i, 1:start + 1]
        return hard_pred

    def _prepare_input_for_ner(self, batch_answers: List[List[str]], batch_types: List[List[int]], separator="; ") \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Важно! Длина ответов с сепараторами не должна быть больше ограничения модели на входные токены
        (в каждом отдельном примере)

        :param batch_answers: списки сгенерированных ответов для каждого примера
        :param batch_types: списки типов сгенерированных ответов
        :param separator: разделитель для сгенерированных ответов в одном примере
        :return: входной тензор B x L' для NER модели, тензор B x L' с разметкой,
                 списки границ сгенерированных ответов для каждого примера
        """
        tokenized_answers = []
        labels_sets = []
        for answers, types in zip(batch_answers, batch_types):
            answers = [answer if answer.strip() else '#' for answer in answers]
            answers = sum([[answer, separator] for answer in answers], [])
            types = sum([[t, -1] for t in types], [])
            _, tokens, labels = self.ner_tokenizer(answers, types)
            tokens, labels = tokens[0], labels[0]
            tokenized_answers.append(tokens)
            labels_sets.append(labels)
        # Padding
        max_len = max(map(len, tokenized_answers))
        token_pad_id = self.ner_tokenizer.word2index[self.ner_tokenizer.pad_token]
        batch_token_ids = torch.full((len(tokenized_answers), max_len), token_pad_id,
                                     dtype=torch.long, device=self.device)
        batch_label_ids = torch.full((len(labels_sets), max_len), self.ner_tokenizer.pad_id,
                                     dtype=torch.long, device=self.device)
        for i, (token_input, labels) in enumerate(zip(tokenized_answers, labels_sets)):
            batch_token_ids[i, :len(token_input)] = torch.tensor(token_input)
            batch_label_ids[i, :len(labels)] = torch.tensor(labels)

        return batch_token_ids, batch_label_ids

    @staticmethod
    def freeze_params(model: nn.Module, reverse=False):
        for param in model.parameters():
            param.requires_grad = reverse

    def update_checkpoint(self, path_to_checkpoint: str, new_name: str = None):
        checkpoint = torch.load(path_to_checkpoint)
        old_model_state = checkpoint['state_dict']
        new_model_state = self.state_dict()
        for k, v in new_model_state.items():
            if k in old_model_state:
                new_model_state[k] = v
        self.load_from_checkpoint()
        torch.save(new_model_state, Path(path_to_checkpoint).with_name(new_name + '.ckpt'))

