import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from anonymization.ref_book import ReferenceBookAnonymization
from anonymization.donated_dataset import DonatedDatasetAnonymization
from datasets.ner_dataset import get_ner_dataset
from models.bert_model import PretrainedBertNER
from utils.log_reader import TensorBoardReader

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    anon_config = yaml.load(open("configs/ref_book_anonymization_config.yaml", 'r'), Loader=yaml.Loader)
    donor_data_config = yaml.load(open("configs/i2b2-2014_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert-large_model_config.yaml", 'r'), Loader=yaml.Loader)
    data_config = yaml.load(open("configs/i2b2-2006_data_config.yaml", 'r'), Loader=yaml.Loader)
    data_config["pretrained_tokenizer"] = model_config.get("pretrained_tokenizer", data_config["pretrained_tokenizer"])
    data_config["max_token_number"] = model_config.get("max_token_number", data_config["max_token_number"])
    # Data processing
    # anonymization = ReferenceBookAnonymization(**anon_config, other_label=data_config['other_label'])
    path_to_donor = Path(donor_data_config["train_data_path"]).with_suffix(".pkl")
    anonymization = DonatedDatasetAnonymization.use_saved_dataset_as_donor(str(path_to_donor),
                                                                           other_label=data_config['other_label'])
    train_dataset = get_ner_dataset(path_to_folder=data_config["train_data_path"],
                                    anonymization=anonymization, **data_config)
    val_dataset = get_ner_dataset(path_to_folder=data_config["validate_data_path"], **data_config)
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
