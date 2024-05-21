import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from datasets.ner_dataset import get_ner_dataset
from anonymization.gpt2_generation import GPT2GenerationAnonymization
from models.bert_model import PretrainedBertNER
from models.gpt2_model import PretrainedGPT2TextInfilling
from utils.log_reader import TensorBoardReader
from mask.personal_entity import MaskEntityType


if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    anon_config = yaml.load(open("configs/gpt2_anonymization_config.yaml", 'r'), Loader=yaml.Loader)
    data_config = yaml.load(open("configs/i2b2-2014_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert-large_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    model_reader = TensorBoardReader(Path(anon_config["log_dir"]) / Path("lightning_logs"))
    path_to_checkpoint = model_reader.get_ckpt_path(anon_config["model_version"])
    text_infill_model = PretrainedGPT2TextInfilling.load_from_checkpoint(path_to_checkpoint, strict=False).to("cuda:0")
    text_infill_model.eval()
    anonymization = GPT2GenerationAnonymization(text_infill_model,
                                                label2type=lambda x: MaskEntityType[x.upper()],
                                                mask_types=list(MaskEntityType), **anon_config)
    train_dataset = get_ner_dataset(path_to_folder=data_config["validate_data_path"],
                                    anonymization=anonymization, device='cpu', **data_config)
    val_dataset = get_ner_dataset(path_to_folder=data_config["train_data_path"], device='cpu', **data_config)
    print(f"Len of train dataset: {len(train_dataset)}\nLen of validation dataset: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=data_config["batch_size"],
                                  collate_fn=train_dataset.get_collate_fn(),
                                  num_workers=10,
                                  pin_memory=False,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                batch_size=data_config["batch_size"],
                                collate_fn=val_dataset.get_collate_fn(),
                                num_workers=10,
                                pin_memory=False,
                                persistent_workers=True)
    # Pytorch lightning
    ner_model = PretrainedBertNER(encoder_vocab_size=len(train_dataset.tokenizer.index2word),
                                  num_classes=len(train_dataset.index2label),
                                  total_steps=model_config["epochs"] * len(train_dataloader),
                                  other_index=train_dataset.label2index[train_dataset.other_label],
                                  pad_index=train_dataset.label2index[train_dataset.pad_label],
                                  **model_config)
    print(ner_model)
    ner_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}', monitor='val_recall', mode='max', save_top_k=1)
    early_stopping_callback = early_stopping.EarlyStopping(monitor="val_recall", patience=5, mode='max')
    trainer_args = {
        "accelerator": "gpu",
        "log_every_n_steps": 1,
        "max_epochs": model_config["epochs"],
        "default_root_dir": model_config["log_dir"],
        "callbacks": [ner_checkpoint_callback, early_stopping_callback]
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.fit(ner_model, train_dataloader, val_dataloader)
    # Plot graphics
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    t_reader.plot_ner_tensorboard_graphics()
