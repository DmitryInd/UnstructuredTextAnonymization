import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from datasets.ner_dataset import get_ner_dataset
from models.bert_model import PretrainedBertNER
from utils.log_reader import TensorBoardReader

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    data_config = yaml.load(open("configs/i2b2_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    train_dataset = get_ner_dataset(data_type=data_config["train_data_type"],
                                    path_to_folder=data_config["train_data_path"],
                                    is_uncased=data_config["is_uncased"],
                                    pretrained_tokenizer=data_config["pretrained_tokenizer_path"],
                                    max_length=data_config["max_token_number"],
                                    eq_max_padding=data_config["eq_max_padding"])
    val_dataset = get_ner_dataset(data_type=data_config["val_data_type"],
                                  path_to_folder=data_config["validate_data_path"],
                                  is_uncased=data_config["is_uncased"],
                                  pretrained_tokenizer=data_config["pretrained_tokenizer_path"],
                                  max_length=data_config["max_token_number"],
                                  eq_max_padding=data_config["eq_max_padding"])
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=data_config["batch_size"],
                                  collate_fn=train_dataset.get_collate_fn())
    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                batch_size=data_config["batch_size"],
                                collate_fn=val_dataset.get_collate_fn())
    # Pytorch lightning
    ner_model = PretrainedBertNER(pretrained_name=model_config["pretrained_model_path"],
                                  encoder_vocab_size=len(train_dataset.tokenizer.index2word),
                                  num_classes=len(train_dataset.index2label),
                                  lr=model_config["lr"],
                                  total_steps=model_config["epochs"] * len(train_dataloader),
                                  adaptation_epochs=model_config["adaptation_epochs"],
                                  div_factor=model_config["div_factor"],
                                  other_index=train_dataset.label2index[data_config["other_label"]],
                                  pad_index=train_dataset.label2index[train_dataset.pad_label])
    print(ner_model)
    ner_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}', monitor='val_recall', mode='max', save_top_k=1)
    trainer_args = {
        "accelerator": "gpu",
        "max_epochs": model_config["epochs"],
        "default_root_dir": model_config["log_dir"],
        "callbacks": ner_checkpoint_callback
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.fit(ner_model, train_dataloader, val_dataloader)
    # Plot graphics
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    t_reader.plot_tensorboard_graphics()
