import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from torch.utils.data import DataLoader
from transformers import set_seed

from datasets.ner_tokenization import WordPieceNERTokenizer

sys.path.insert(1, "./src")
from datasets.text_infill_dataset import get_text_infill_dataset
from models.gpt2_model import PretrainedGPT2TextInfilling
from models.bert_model import PretrainedBertNER
from utils.log_reader import TensorBoardReader
from mask.personal_entity import MaskEntityType


if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    train_data_config = yaml.load(open("configs/i2b2-2014_data_config.yaml", 'r'), Loader=yaml.Loader)
    train_data_config["pretrained_tokenizer"] = "data/tokenizer/official_gpt2_encoder"
    validate_data_config = yaml.load(open("configs/i2b2-2014_data_config.yaml", 'r'), Loader=yaml.Loader)
    validate_data_config["pretrained_tokenizer"] = "data/tokenizer/official_gpt2_encoder"
    model_config = yaml.load(open("configs/gpt2_fine-tune_config.yaml", 'r'), Loader=yaml.Loader)
    ner_model_config = yaml.load(open("configs/bert-large_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    train_dataset = get_text_infill_dataset(split="train", path_to_data=train_data_config["train_data_path"],
                                            label2type=lambda x: MaskEntityType[x.upper()],
                                            mask_types=list(MaskEntityType),
                                            **train_data_config, device='cpu')
    val_dataset = get_text_infill_dataset(split="validate", path_to_data=validate_data_config["validate_data_path"],
                                          label2type=lambda x: MaskEntityType[x.upper()],
                                          mask_types=list(MaskEntityType),
                                          **validate_data_config, device='cpu')
    print(f"Len of train dataset: {len(train_dataset)}\nLen of validation dataset: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=model_config["batch_size"],
                                  collate_fn=train_dataset.get_collate_fn(),
                                  num_workers=10,
                                  pin_memory=False,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                batch_size=model_config["batch_size"],
                                collate_fn=val_dataset.get_collate_fn(),
                                num_workers=10,
                                pin_memory=False,
                                persistent_workers=True)
    # Pytorch lightning
    ner_reader = TensorBoardReader(Path(ner_model_config["log_dir"]) / Path("lightning_logs"))
    ner_model = PretrainedBertNER.load_from_checkpoint(ner_reader.get_ckpt_path(model_config["ner_model_version"]))
    ner_model.eval()
    model_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    path_to_checkpoint = model_reader.get_ckpt_path(model_config["model_version"])
    text_infill_model = PretrainedGPT2TextInfilling.load_from_checkpoint(
        path_to_checkpoint,
        strict=False,
        total_steps=model_config["epochs"] * len(train_dataloader),
        num_classes=ner_model.num_classes,  # It was at this moment that I knew... pad label is superfluous
        **model_config
    )
    # Set additional parameters
    text_infill_model.tokenizer = train_dataset.tokenizer
    text_infill_model.ner_model = ner_model
    text_infill_model.ner_tokenizer = WordPieceNERTokenizer([[]], -1,
                                                            pad_flag=True,
                                                            max_sent_len=ner_model_config["max_token_number"],
                                                            overlap=0,
                                                            pretrained_name=ner_model_config["pretrained_tokenizer"])
    print(text_infill_model)
    if model_config["step_type"] == "classic":
        text_infill_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}', monitor='val_loss',
                                                          mode='min', save_top_k=1)
        early_stopping_callback = early_stopping.EarlyStopping(monitor="val_loss", patience=5, mode='min')
    elif model_config["step_type"] == "rl":
        text_infill_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}', monitor='val_reward',
                                                          mode='max', save_top_k=1)
        early_stopping_callback = early_stopping.EarlyStopping(monitor="val_reward", patience=5, mode='max')
    else:
        raise ValueError(f"Invalid step_type: {model_config['step_type']}")
    trainer_args = {
        "accelerator": "gpu",
        "log_every_n_steps": 1,
        "gradient_clip_val": 100,
        "max_epochs": model_config["epochs"],
        "default_root_dir": model_config["log_dir"],
        "callbacks": [text_infill_checkpoint_callback, early_stopping_callback]
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.fit(text_infill_model, train_dataloader, val_dataloader)
    # Plot graphics
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    t_reader.plot_text_infill_tensorboard_graphics()
