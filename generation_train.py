import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, early_stopping
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from datasets.text_infill_dataset import get_text_infill_dataset
from models.gpt2_model import PretrainedGPT2TextInfilling
from utils.log_reader import TensorBoardReader
from mask.n_gram import RandomNgramsMaskFn
from mask.personal_entity import MaskEntityType

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    mask_config = yaml.load(open("configs/ngram_mask_config.yaml", 'r'), Loader=yaml.Loader)
    data_config = yaml.load(open("configs/roc_stories_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/gpt2-small_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    masker = RandomNgramsMaskFn(mask_types=list(MaskEntityType), **mask_config)
    train_dataset = get_text_infill_dataset(path_to_data=data_config["train_data_path"], split="train", masker=masker,
                                            **data_config, device='cpu')
    val_dataset = get_text_infill_dataset(path_to_data=data_config["validate_data_path"], split="valid", masker=masker,
                                          **data_config, device='cpu')
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
    text_infill_model = PretrainedGPT2TextInfilling(
        total_steps=model_config["epochs"] * len(train_dataloader),
        vocab_size=train_dataset.tokenizer.vocab_size,
        end_infill_id=train_dataset.tokenizer.end_infill_id,
        **model_config
    )
    text_infill_model.tokenizer = train_dataset.tokenizer
    print(text_infill_model)
    text_infill_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}', monitor='val_loss',
                                                      mode='min', save_top_k=1)
    early_stopping_callback = early_stopping.EarlyStopping(monitor="val_loss", patience=5, mode='min')
    trainer_args = {
        "accelerator": "gpu",
        "log_every_n_steps": 1,
        "max_epochs": model_config["epochs"],
        "default_root_dir": model_config["log_dir"],
        "callbacks": [text_infill_checkpoint_callback, early_stopping_callback]
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.fit(text_infill_model, train_dataloader, val_dataloader)
    # Plot graphics
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    t_reader.plot_text_infill_tensorboard_graphics()
