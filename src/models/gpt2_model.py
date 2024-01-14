import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Recall, Precision, F1Score
import torch.nn.functional as F
from datasets.tokenization import TargetType
from transformers import GPT2LMHeadModel


class PretrainedGPT2TextInfilling(pl.LightningModule):
    def __init__(self, pretrained_name: str, vocab_size: int,
                 lr: float, total_steps: int, adaptation_part: int, div_factor: int,
                 other_index: int, pad_index: int):
        """
        :param pretrained_name: название предобученной GPT2 модели из hugging face hub
        :param vocab_size: итоговый размер словаря (с добавлением или удалением части токенов)
        :param lr: максимальный learning rate
        :param total_steps: полное количество шагов обучения: ~ кол-во эпох * размер батча
        :param adaptation_part: доля эпох для обновления весов с низким learning rate
        :param div_factor: максимальный делитель, на который уменьшается learning rate в OneCycle подходе
        :param other_index: индекс метки обычного слова, правильная классификация которого не столь важна
        :param pad_index: индекс метки pad токена, которая не должна учитываться
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_name)
        self.other_index = other_index
        self.pad_index = pad_index
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_index)
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        self.adaptation_part = adaptation_part
        # Metrics
        # self.train_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        # self.val_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=pad_index)

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.head(x)  # B, L, C
        return x

    def configure_optimizers(self):
        # Similar to the article
        params = list(self.parameters())
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
            anneal_strategy='linear',
            final_div_factor=self.div_factor
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # TODO
        inputs, tts = batch
        labels_context = self.tts_to_labels(inputs, tts, [TargetType.CONTEXT])
        labels_infill = self.tts_to_labels(inputs, tts,
                                      [TargetType.INFILL, TargetType.INFILL_SPECIAL])
        logits, _ = self.model(inputs)
        logits_relevant = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
        loss_context = F.cross_entropy(
            logits_relevant,
            labels_context[:, 1:].contiguous().view(-1),
            ignore_index=-1)
        loss_infill = F.cross_entropy(
            logits_relevant,
            labels_infill[:, 1:].contiguous().view(-1),
            ignore_index=-1)

        loss_context_item = loss_context.item()
        loss_infill_item = loss_infill.item()

        _, x, y = batch  # X: B, L
        predictions = self(x).transpose(2, 1)  # B, L, C -> B, C, L
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-2)
        # self.train_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        # self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        predictions = self(x).transpose(2, 1)  # B, L, C -> B, C, L
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-2)
        # self.val_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        # self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, x, y = batch
        hard_pred = torch.argmax(self(x).transpose(2, 1), dim=-2)
        # self.val_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        # self.log('test_recall', self.val_recall, on_step=False, on_epoch=True, logger=True)

    @staticmethod
    def tts_to_labels(inputs, tts, label_tts):
        selector = torch.zeros_like(inputs, dtype=torch.bool)
        for tt in label_tts:
            selector |= tts == tt.value
        return torch.where(
            selector,
            inputs,
            torch.full_like(inputs, -1))

    @staticmethod
    def freeze_params(model: nn.Module, reverse=False):
        for param in model.parameters():
            param.requires_grad = reverse
