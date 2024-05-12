from typing import Optional

import torch
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate
from torch import nn
from datasets.text_infill_tokenization import TextInfillTokenizer, TargetType
from transformers import GPT2LMHeadModel


class PretrainedGPT2TextInfilling(pl.LightningModule):
    def __init__(self, pretrained_name: str, vocab_size: int, train_context,
                 lr: float, total_steps: int, adaptation_part: int, div_factor: int,
                 end_infill_id=None, **kwargs):
        """
        :param pretrained_name: название предобученной GPT2 модели из hugging face hub
        :param vocab_size: итоговый размер словаря (с добавлением или удалением части токенов)
        :param train_context: учитывать ли лосс по контексту
        :param lr: максимальный learning rate
        :param total_steps: полное количество шагов обучения: ~ кол-во эпох * размер батча
        :param adaptation_part: доля эпох для обновления весов с низким learning rate
        :param div_factor: максимальный делитель, на который уменьшается learning rate в OneCycle подходе
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        self.adaptation_part = adaptation_part
        self.train_context = train_context
        self.end_infill_id = end_infill_id
        # Metrics for quality evaluation
        self.tokenizer: Optional[TextInfillTokenizer] = None
        self.train_cer = CharErrorRate()
        self.val_cer = CharErrorRate()

    def forward(self, x):
        x = self.model(x)  # B, L, C
        return x

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

    def training_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        labels_context = self.tts_to_labels(inputs, tts, [TargetType.CONTEXT])
        labels_infill = self.tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])
        logits = self.forward(inputs[:, :-1]).logits
        logits = logits.transpose(2, 1)  # B, L, C -> B, C, L
        loss_context = self.criterion(logits, labels_context[:, 1:])
        loss_infill = self.criterion(logits, labels_infill[:, 1:])
        loss = loss_infill
        if self.train_context:
            loss += loss_context
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if self.tokenizer is not None:
            hard_pred = torch.argmax(logits, dim=-2)
            for orig, gen in zip(inputs, hard_pred):
                self.train_cer.update(self.tokenizer.parse_answers(gen, orig),
                                      self.tokenizer.parse_answers(orig))
        self.log('train_cer', self.train_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, inputs, tts = batch  # B, L
        labels_context = self.tts_to_labels(inputs, tts, [TargetType.CONTEXT])
        labels_infill = self.tts_to_labels(inputs, tts, [TargetType.INFILL, TargetType.INFILL_SPECIAL])
        logits = self.forward(inputs[:, :-1]).logits
        logits = logits.transpose(2, 1)  # B, L, C -> B, C, L
        loss_context = self.criterion(logits, labels_context[:, 1:])
        loss_infill = self.criterion(logits, labels_infill[:, 1:])
        loss = loss_infill
        if self.train_context:
            loss += loss_context
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if self.tokenizer is not None:
            hard_pred = torch.argmax(logits, dim=-2)
            for orig, gen in zip(inputs, hard_pred):
                self.val_cer.update(self.tokenizer.parse_answers(gen, orig),
                                    self.tokenizer.parse_answers(orig))
        self.log('val_cer', self.val_cer, on_step=False, on_epoch=True, logger=True, prog_bar=True)

    @torch.no_grad()
    def inference(self, inputs: torch.Tensor, tts: torch.Tensor):
        masks_number = (tts == TargetType.CONTEXT_SPECIAL.value).sum(dim=1)
        positions = (tts != TargetType.PAD.value).sum(dim=1)
        finished = set()
        while len(finished) < inputs.shape[0]:
            logits = self.forward(inputs).logits  # B, L, C
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
    def tts_to_labels(inputs, tts, label_tts):
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

    @staticmethod
    def freeze_params(model: nn.Module, reverse=False):
        for param in model.parameters():
            param.requires_grad = reverse
