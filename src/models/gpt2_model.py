import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import grad_norm
from torch import nn
from torch.nn import Sequential, Linear, LeakyReLU
from torch.nn.functional import log_softmax, gumbel_softmax, softmax, normalize
from torchmetrics import Recall, Metric
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text import CharErrorRate
from transformers import GPT2LMHeadModel

from datasets.ner_tokenization import NERTokenizer
from datasets.text_infill_tokenization import TextInfillTokenizer, TargetType
from models.bert_model import PretrainedBertNER


class PretrainedGPT2TextInfilling(pl.LightningModule):
    def __init__(self, pretrained_name: str, vocab_size: int, train_context: float,
                 lr: float, total_steps: int, adaptation_part: int, div_factor: int, step_type='classic',
                 types_weights=None, end_infill_id=None, target_types_pred=0.0, real_types_match=0.0, num_classes=2,
                 sample_temperature=0., self_critical=False, with_context=False, maximize_distance=0.0,
                 sameness_penalty=0.0, sameness_threshold=0.01, **kwargs):
        """
        :param pretrained_name: название предобученной GPT2 модели из hugging face hub
        :param vocab_size: итоговый размер словаря (с добавлением или удалением части токенов)
        :param train_context: вес функции потерь на контексте (0. - отключение)
        :param lr: максимальный learning rate
        :param total_steps: полное количество шагов обучения: ~ кол-во эпох * размер батча
        :param adaptation_part: доля эпох для обновления весов с низким learning rate
        :param div_factor: максимальный делитель, на который уменьшается learning rate в OneCycle подходе
        :param step_type: тип шага (classic / rl)
        :param types_weights: веса типов генерируемых данных для кросс-энтропии
        :param end_infill_id: id end_infill токена для инференса модели
        :param target_types_pred: вес функции потерь для предсказания целевых меток для сгенерированных данных
        :param real_types_match: вес функции потерь для предсказания истинных меток для сгенерированных данных
        :param num_classes: количество типов замаскированных данных, генерируемых моделью (num_classes > max_type_id)
        :param sample_temperature: температура для семплирования при использовании RL
        :param self_critical: использовать ли подход Self-critical Sequence Training для bias
        :param with_context: использовать ли контекст при оценке генерации с помощью NER
        :param maximize_distance: вес награды за дальность сгенерированных данных от оригинальных ответов
        :param sameness_penalty: вес штрафа за одинаковые ответы от модели
        :param sameness_threshold: порог однообразия, с которого начисляется штраф
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
        self.step_type = step_type
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        self.adaptation_part = adaptation_part
        # Parameters for classic step
        if not isinstance(types_weights, torch.Tensor) and types_weights is not None:
            types_weights = torch.tensor(types_weights)
        self.g_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.t_criterion = nn.CrossEntropyLoss(weight=types_weights,
                                               reduction='mean', ignore_index=-1)
        self.gt_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.gt_counter = torch.zeros((num_classes, ), dtype=torch.int) if types_weights is not None else None
        self.gt_alpha = 0.9
        self.train_context = train_context
        self.target_types_pred = target_types_pred
        self.real_types_match = real_types_match
        # Parameters for RL
        self.rl_g_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.rl_t_criterion = nn.CrossEntropyLoss(weight=types_weights,
                                                  reduction='none', ignore_index=-1)
        self.sample_temperature = sample_temperature
        self.self_critical = self_critical
        self.with_context = with_context
        self.maximize_distance = maximize_distance
        self.sameness_penalty = sameness_penalty
        self.sameness_threshold = sameness_threshold
        # Additional modules for training
        self.tokenizer: Optional[TextInfillTokenizer] = None
        self.ner_model: Optional[PretrainedBertNER] = None
        self.ner_tokenizer: Optional[NERTokenizer] = None
        self.ner_replace: Optional[re.Pattern] = None
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

    def classic_step(self, inputs: torch.Tensor, tts: torch.Tensor,
                     target_type_recall: Metric, real_type_recall: Metric, cer: CharErrorRate) -> torch.Tensor:
        target_context = self.tts_to_targets(inputs, tts, [TargetType.CONTEXT])
        target_infill = self.tts_to_targets(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])
        logits, types_logits = self.forward(inputs[:, :-1], gen_types=self.target_types_pred > 0.)
        logits, types_logits = logits.transpose(2, 1), types_logits.transpose(2, 1)  # B, L, C -> B, C, L
        # Get fair predictions for answer and their types
        answers_starts = (torch.nonzero(tts == TargetType.CONTEXT_INFILL_SEP.value)[:, 1] + 1).tolist()
        with torch.no_grad():
            generated, _ = self.inference(inputs, tts,
                                          temperature=self.sample_temperature if self.training else 0.,
                                          fresh_start=True)
        # Compute main generation loss
        loss_infill = self.g_criterion(logits, target_infill[:, 1:])
        loss = loss_infill
        if self.train_context:
            loss_context = self.g_criterion(logits, target_context[:, 1:])
            loss += self.train_context * loss_context
        # Compute target labels loss
        if self.target_types_pred and self.tokenizer is not None:
            target_types_list = inputs[tts == TargetType.CONTEXT_SPECIAL.value].cpu()
            target_types_list = np.vectorize(lambda x: self.tokenizer.id_to_mask_type[x].value)(
                target_types_list).tolist()
            target_types = self.tokenizer.mark_up_types(inputs, target_types_list)[:, :-1]
            loss += self.target_types_pred * self.t_criterion(types_logits, target_types)
            target_type_recall.update(torch.argmax(types_logits, dim=-2), target_types)
        # Compute real labels loss
        if (self.real_types_match and self.tokenizer is not None
                and self.ner_model is not None and self.ner_tokenizer is not None):
            assert self.ner_tokenizer.pad_id == -1, "Pad id for NER tokenizer must be -1"
            assert self.ner_tokenizer.overlap == 0, "Overlap of NER tokenizer must be 0"
            _, g_types_logits = self.forward(torch.maximum(generated[:, :-1], torch.tensor(0)), gen_types=True)
            g_types_logits = g_types_logits.transpose(2, 1)  # B, L, C -> B, C, L
            sample_answers = [self.tokenizer.parse_answers(gen, st) for st, gen in zip(answers_starts, generated)]
            num_answers = [len(answers) for answers in sample_answers]
            pseudo_labels = [list(range(1, num + 1)) for num in num_answers]
            ner_input, ner_labels = self._prepare_input_for_ner(sample_answers, pseudo_labels)
            with torch.no_grad():
                log_probs = log_softmax(self.ner_model(ner_input), dim=-1)
            real_types_list = self.ner_tokenizer.parse_labels_from_marked_up_log_probs(log_probs, ner_labels,
                                                                                       num_answers)
            real_types = self.tokenizer.mark_up_types(generated, real_types_list)[:, :-1]
            if self.gt_counter is not None:
                self.gt_counter = self._counter_update(inputs, real_types, self.gt_counter, self.gt_alpha)
                self.gt_criterion.weight = normalize(1 / self.gt_counter, p=1, dim=0)
            loss += self.real_types_match * self.gt_criterion(g_types_logits, real_types)
            real_type_recall.update(torch.argmax(g_types_logits, dim=-2), real_types)
        # Compute cer statistics
        if self.tokenizer is not None:
            answers_numbers = ((tts == TargetType.INFILL_SPECIAL.value).sum(dim=-1) - 1).tolist()
            for answers_start, answers_number, orig, gen in zip(answers_starts, answers_numbers, inputs, generated):
                cer.update(self.tokenizer.parse_answers(gen, answers_start, answers_number),
                           self.tokenizer.parse_answers(orig, answers_start))
        return loss

    def rl_policy_gradient_step(self, inputs: torch.Tensor, tts: torch.Tensor, cer: CharErrorRate) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None and self.ner_tokenizer is not None and self.ner_model is not None
        assert self.ner_tokenizer.pad_id == -1, "Pad id for NER tokenizer must be -1"
        assert self.ner_tokenizer.overlap == 0, "Overlap of NER tokenizer must be 0"

        with torch.no_grad():
            predictions, _ = self.inference(inputs, tts,
                                            temperature=self.sample_temperature if self.training else 0.,
                                            fresh_start=True)
        if self.self_critical:
            with torch.no_grad():
                greedy_pred, _ = self.inference(inputs, tts, fresh_start=True)
        logits = self.forward(torch.maximum(predictions[:, :-1], torch.tensor(0)))[0].transpose(1, 2)
        target_infill = self.tts_to_targets(
            predictions, tts,
            [TargetType.CONTEXT, TargetType.CONTEXT_SPECIAL, TargetType.CONTEXT_INFILL_SEP], reverse=True
        )[:, 1: logits.size(2) + 1]
        # Compute the log probabilities of the current states
        state_log_prob = self.rl_g_criterion(logits, target_infill).sum(dim=-1) / (target_infill != -1).sum(dim=-1)
        # Compute distance between original and predicted answers
        local_cer = []
        greedy_cer = []
        answers_starts = (torch.nonzero(tts == TargetType.CONTEXT_INFILL_SEP.value)[:, 1] + 1).tolist()
        answers_numbers = ((tts == TargetType.INFILL_SPECIAL.value).sum(dim=-1) - 1).tolist()
        for i in range(predictions.size(0)):
            gen_answers = self.tokenizer.parse_answers(predictions[i], answers_starts[i], answers_numbers[i])
            orig_answers = self.tokenizer.parse_answers(inputs[i], answers_starts[i])
            cer.update(gen_answers, orig_answers)
            if self.maximize_distance:
                local_cer.append(char_error_rate(gen_answers, orig_answers).item())
            if self.maximize_distance and self.self_critical:
                greedy_answers = self.tokenizer.parse_answers(greedy_pred[i], answers_starts[i], answers_numbers[i])
                greedy_cer.append(char_error_rate(greedy_answers, orig_answers).item())
        # Compute reward
        reward = self._compute_reward(predictions, tts, local_cer)
        if self.self_critical:
            reward -= self._compute_reward(greedy_pred, tts, greedy_cer)
        # Compute loss from reward
        loss = torch.mean(reward * state_log_prob)
        # Estimate the sameness of generation
        # entropy = state_log_prob.mean()  # B -> 1
        entropy = -(log_softmax(logits, dim=-2) * softmax(logits, dim=-2)).sum(dim=-2)  # B, C, L -> B, L
        entropy = entropy.where(target_infill != -1, 0.).sum(dim=-1) / (target_infill != -1).sum(dim=-1)  # B, L -> B
        entropy = torch.clamp_max(entropy, self.sameness_threshold).mean()  # B -> 1
        if self.sameness_penalty:
            loss -= self.sameness_penalty * entropy
        return loss, reward.mean(), entropy

    def _compute_reward(self, predictions: torch.Tensor, tts: torch.Tensor, local_cer: Optional[List[int]] = None):
        # Get hard predictions for answer and their types
        answers_starts = (torch.nonzero(tts == TargetType.CONTEXT_INFILL_SEP.value)[:, 1] + 1).tolist()
        masks_num = (tts == TargetType.CONTEXT_SPECIAL.value).sum(dim=-1).tolist()
        sample_answers = [self.tokenizer.parse_answers(gen, st, n)
                          for st, n, gen in zip(answers_starts, masks_num, predictions)]
        # Get target types of infilled masks
        labels = []
        target_types_list = predictions[tts == TargetType.CONTEXT_SPECIAL.value].cpu()
        target_types_list = np.vectorize(lambda x: self.tokenizer.id_to_mask_type[x].value)(target_types_list).tolist()
        cursor = 0
        for n in masks_num:
            labels.append(target_types_list[cursor:cursor + n])
            cursor += n
        # Get real types of infilled masks
        if self.with_context:
            context = self.tokenizer.parse_context(predictions, tts, answers_starts)
            ner_input, ner_labels = self._prepare_input_for_ner(sample_answers, labels, separator=' ', context=context)
        else:
            ner_input, ner_labels = self._prepare_input_for_ner(sample_answers, labels)
        with torch.no_grad():
            ner_logits = self.ner_model(ner_input).transpose(1, 2)  # B, L, C -> B, C, L
        # B, C, L -> B, L -> B; (-inf; 0]
        reward = -self.rl_t_criterion(ner_logits, ner_labels).sum(dim=-1) / (ner_labels != -1).sum(dim=-1)
        if self.maximize_distance:
            reward += self.maximize_distance * torch.tensor(local_cer, device=self.device)
        # if self.sameness_penalty:
        #     reward -= self.sameness_penalty *
        #               torch.clamp_min(self.sameness_threshold / (state_log_prob + 1e-6) - 1.0, 0.0)
        return reward

    def training_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        if self.step_type == 'classic':
            loss = self.classic_step(inputs, tts,
                                     self.train_target_type_recall, self.train_real_type_recall, self.train_cer)
            self.log('train_target_type_recall', self.train_target_type_recall, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
            self.log('train_real_type_recall', self.train_real_type_recall, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
        elif self.step_type == 'rl':
            loss, reward, entropy = self.rl_policy_gradient_step(inputs, tts, self.train_cer)
            self.log('train_reward', reward.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
            self.log('train_entropy', entropy.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        else:
            raise ValueError(f'Unrecognized step_type: {self.step_type}')

        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_cer', self.train_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        if self.step_type == 'classic':
            loss = self.classic_step(inputs, tts,
                                     self.val_target_type_recall, self.val_real_type_recall, self.val_cer)
            self.log('val_target_type_recall', self.val_target_type_recall, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
            self.log('val_real_type_recall', self.val_real_type_recall, on_step=False, on_epoch=True,
                     logger=True, prog_bar=True)
        elif self.step_type == 'rl':
            loss, reward, entropy = self.rl_policy_gradient_step(inputs, tts, self.val_cer)
            self.log('val_reward', reward.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
            self.log('val_entropy', entropy.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        else:
            raise ValueError(f'Unrecognized step_type: {self.step_type}')

        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_cer', self.val_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        grads_norm = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('grads_norm', grads_norm, on_step=True, on_epoch=False, logger=True, prog_bar=True)

    def inference(self, inputs: torch.Tensor, tts: torch.Tensor, temperature=0.0, fresh_start=False):
        predictions = inputs.detach().clone()  # B, L
        logits = []
        masks_number = (tts == TargetType.CONTEXT_SPECIAL.value).sum(dim=1)
        if fresh_start:
            start_positions = torch.nonzero(tts == TargetType.CONTEXT_INFILL_SEP.value)[:, 1]
        else:
            start_positions = (tts != TargetType.PAD.value).sum(dim=1) - 1
        l_border = start_positions.min().item()
        predictions[:, l_border + 1:] = -1
        past_key_values = None
        finished = set()
        while len(finished) < inputs.shape[0]:
            shift = l_border if past_key_values is not None else 0
            results = self.model(torch.maximum(predictions[:, shift: l_border + 1], torch.tensor(0)),
                                 past_key_values=past_key_values)
            logits.append(results.logits)
            if temperature:
                hard_pred = torch.argmax(gumbel_softmax(results.logits, tau=temperature), dim=-1)  # B, L, C
            else:
                hard_pred = torch.argmax(results.logits, dim=-1)  # B, L, C
            for i, row in enumerate(hard_pred):
                if start_positions[i] > l_border:
                    predictions[i, l_border + 1] = inputs[i, l_border + 1]
                    continue
                if (l_border + 1) >= inputs.shape[1] or masks_number[i] <= 0:
                    finished.add(i)
                    continue
                if row[l_border - shift] == self.end_infill_id:
                    masks_number[i] -= 1
                predictions[i, l_border + 1] = row[l_border - shift]
            l_border += 1
            past_key_values = results.past_key_values
        return predictions, torch.cat(logits, dim=1)

    @staticmethod
    def tts_to_targets(inputs, tts, label_tts, reverse=False):
        """
        Заменяет нецелевые токены в inputs на -1
        """
        selector = torch.zeros_like(inputs, dtype=torch.bool)
        for tt in label_tts:
            selector |= (tts == tt.value)
        if reverse:
            selector = torch.logical_not(selector)
        return torch.where(
            selector,
            inputs,
            torch.full_like(inputs, -1))

    @staticmethod
    def _get_hard_prediction(self, logits: torch.Tensor, inputs: torch.Tensor | None,
                             answers_starts: List[int] | None = None):
        assert len(inputs.shape) == 2 and len(logits.shape) == 3, \
            "The shapes of inputs and logits must be BxL and BxCx(L-1)"
        assert inputs.shape[0] == logits.shape[0] and inputs.shape[1] - 1 == logits.shape[2], \
            "The shapes of inputs and logits do not match"

        hard_pred = torch.argmax(logits, dim=-2)
        if inputs is not None and answers_starts is not None:
            for i, start in enumerate(answers_starts):
                hard_pred[i, :start - 1] = inputs[i, 1:start]
        return hard_pred

    def _counter_update(self, types, batch_size, old_counter, alpha):
        counter = torch.bincount(
            types.where(types >= 0, torch.tensor(len(old_counter))).flatten(),
            minlength=len(old_counter)
        )[:len(old_counter)] + batch_size / 8
        if self.global_step:
            counter = alpha * old_counter.to(self.device) + (1 - alpha) * counter
        return counter

    def _prepare_input_for_ner(self, batch_answers: List[List[str]], batch_types: List[List[int]], separator="; ",
                               context: Optional[List[List[str]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Важно! Длина ответов с сепараторами или контекстом не должна быть больше ограничения модели на входные токены
        (в каждом отдельном примере; примеры, которые длиннее ограничения, будут обрезаны)

        :param batch_answers: списки сгенерированных ответов для каждого примера
        :param batch_types: списки типов сгенерированных ответов
        :param separator: разделитель для сгенерированных ответов в одном примере
        :param context: (опционально) контекст, в который будут помещены ответы ({context[i]}{answers[i]})
        :return: входной тензор B x L' для NER модели, тензор B x L' с разметкой,
                 списки границ сгенерированных ответов для каждого примера
        """
        tokenized_answers = []
        labels_sets = []
        if self.ner_replace is None:
            self.ner_replace = re.compile(fr"[^\s{re.escape(''.join(set(''.join(self.ner_tokenizer.word2index.keys()))))}]")
        for i, (answers, types) in enumerate(zip(batch_answers, batch_types)):
            answers = [self.ner_replace.sub('', answer) for answer in answers]
            answers = [answer if answer.strip() else '#' for answer in answers]
            if context is not None:
                cur_context = [self.ner_replace.sub('', seg) for seg in context[i]]
                cur_context += [separator] * (np.max(len(answers) - len(cur_context), 0))
                answers = (sum([[cont, answer] for answer, cont in zip(answers, cur_context[:len(answers)])], [])
                           + cur_context[len(answers):])
                types = sum([[-1, t] for t in types], []) + [-1] * (len(cur_context) - len(types))
            else:
                answers = sum([[answer, separator] for answer in answers], [])
                types = sum([[t, -1] for t in types], [])
            _, tokens, labels = self.ner_tokenizer(answers or ['#'], types or [-1])
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
