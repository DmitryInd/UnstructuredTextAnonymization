import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Recall, Precision, F1Score
from transformers import BertModel


class PretrainedBertNER(pl.LightningModule):
    def __init__(self, pretrained_name: str, encoder_vocab_size: int,
                 lr: float, total_steps: int, adaptation_epochs: int, div_factor: int,
                 num_classes: int, other_index: int, pad_index: int, **kwargs):
        """
        :param pretrained_name: название предобученного BERT из hugging face hub
        :param encoder_vocab_size: итоговый размер словаря (с добавлением или удалением части токенов)
        :param lr: максимальный learning rate
        :param total_steps: полное количество шагов обучения: ~ кол-во эпох * размер батча
        :param adaptation_epochs: количество эпох для адаптации новых весов к замороженным предобученным пар-рам
        :param div_factor: максимальный делитель, на который уменьшается learning rate в OneCycle подходе
        :param num_classes: количество предсказываемых классов
        :param other_index: индекс метки обычного слова, правильная классификация которого не столь важна
        :param pad_index: индекс метки pad токена, которая не должна учитываться
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BertModel.from_pretrained(pretrained_name)
        self.head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.model.config.hidden_size//2, num_classes)
        )
        self.num_classes = num_classes
        self.other_index = other_index
        self.pad_index = pad_index
        # Expanding or reducing the space of the encoder embeddings
        self.model.resize_token_embeddings(encoder_vocab_size)
        # Parameters of optimization
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_index)
        self.lr = lr
        self.div_factor = div_factor
        self.total_steps = total_steps
        self.adaptation_epochs = adaptation_epochs
        # Metrics
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        self.train_f1_score = F1Score(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, ignore_index=pad_index)
        self.val_f1_score = F1Score(task="multiclass", num_classes=num_classes, ignore_index=pad_index)

    def forward(self, x, encoder_attention_mask=None):
        x = self.model(x, encoder_attention_mask=encoder_attention_mask).last_hidden_state
        x = self.head(x)  # B, L, C
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.lr,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=self.div_factor
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        _, x, y = batch  # X: B, L
        if self.current_epoch < self.adaptation_epochs:
            self.freeze_params(self.model)  # Adaptation of new parameters to pretrained
        else:
            self.freeze_params(self.model, reverse=True)
        padding = self._create_attention_mask(y)
        predictions = self(x, encoder_attention_mask=padding).transpose(2, 1)  # B, L, C -> B, C, L
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-2)
        self.train_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        self.train_precision(hard_pred, y)
        self.train_f1_score(hard_pred, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, logger=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, logger=True)
        self.log('train_f1', self.train_f1_score, on_step=False, on_epoch=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        padding = self._create_attention_mask(y)
        predictions = self(x, encoder_attention_mask=padding).transpose(2, 1)  # B, L, C -> B, C, L
        loss = self.criterion(predictions, y)
        hard_pred = torch.argmax(predictions, dim=-2)
        self.val_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        self.val_precision(hard_pred, y)
        self.val_f1_score(hard_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, logger=True)
        self.log('val_f1', self.val_f1_score, on_step=False, on_epoch=True, logger=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _, x, y = batch
        padding = self._create_attention_mask(y)
        hard_pred = torch.argmax(self(x, encoder_attention_mask=padding).transpose(2, 1), dim=-2)
        self.val_recall(hard_pred, y.where(y != self.other_index, self.pad_index))
        self.val_precision(hard_pred, y)
        self.val_f1_score(hard_pred, y)
        self.log('test_recall', self.val_recall, on_step=False, on_epoch=True, logger=True)
        self.log('test_precision', self.val_precision, on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', self.val_f1_score, on_step=False, on_epoch=True, logger=True)

    @staticmethod
    def freeze_params(model: nn.Module, reverse=False):
        for param in model.parameters():
            param.requires_grad = reverse

    def _create_attention_mask(self, markup: torch.Tensor) -> torch.Tensor:
        # Input:  [CLS] token token ... [EOS] [PAD] [PAD] ...
        # Markup: [PAD] label label ... [PAD] [PAD] [PAD] ...
        attention_mask = markup.ne(self.pad_index).int()
        attention_mask[:, 0] = 1  # For [CLS] token
        attention_mask[(1 - attention_mask).cumsum(dim=-1).eq(1)] = 1  # For [EOS] token
        return attention_mask
