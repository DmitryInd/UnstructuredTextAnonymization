import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from datasets.text_infill_dataset import get_text_infill_dataset
from models.gpt2_model import PretrainedGPT2TextInfilling
from utils.log_reader import TensorBoardReader

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    data_config = yaml.load(open("configs/roc_stories_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/gpt2-small_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    # train_dataset = get_text_infill_dataset(dataset_type=data_config["dataset_type"],
    #                                         path_to_data=data_config["train_data_path"],
    #                                         split="train",
    #                                         other_label=data_config["other_label"],
    #                                         is_uncased=data_config["is_uncased"],
    #                                         pretrained_tokenizer="data/tokenizer/official_gpt2_encoder",
    #                                         max_full_ex_len=data_config["max_token_number"],
    #                                         overlap=data_config["overlap"],
    #                                         device='cpu')

    train_dataset = get_text_infill_dataset(dataset_type=data_config["dataset_type"],
                                            path_to_data=data_config["train_data_path"], split="train",
                                            is_uncased=data_config["is_uncased"],
                                            pretrained_tokenizer="data/tokenizer/official_gpt2_encoder",
                                            max_full_ex_len=data_config["max_full_ex_len"],
                                            overlap=data_config["overlap"], mask_p=data_config["mask_p"],
                                            max_span_len=data_config["max_span_len"],
                                            max_num_examples=data_config["max_num_train_examples"],
                                            num_examples_per_doc=data_config["num_examples_per_doc"],
                                            max_num_retries_per_ex=data_config["max_num_retries_per_ex"],
                                            min_masked_spans_per_ex=data_config["min_masked_spans_per_ex"],
                                            max_masked_spans_per_ex=data_config["max_masked_spans_per_ex"],
                                            device='cpu')

    val_dataset = get_text_infill_dataset(dataset_type=data_config["dataset_type"],
                                          path_to_data=data_config["validate_data_path"], split="valid",
                                          is_uncased=data_config["is_uncased"],
                                          pretrained_tokenizer="data/tokenizer/official_gpt2_encoder",
                                          max_full_ex_len=data_config["max_full_ex_len"],
                                          overlap=data_config["overlap"], mask_p=data_config["mask_p"],
                                          max_span_len=data_config["max_span_len"],
                                          max_num_examples=data_config["max_num_valid_examples"],
                                          num_examples_per_doc=data_config["num_examples_per_doc"],
                                          max_num_retries_per_ex=data_config["max_num_retries_per_ex"],
                                          min_masked_spans_per_ex=data_config["min_masked_spans_per_ex"],
                                          max_masked_spans_per_ex=data_config["max_masked_spans_per_ex"], device='cpu')
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
    text_infill_model = PretrainedGPT2TextInfilling(pretrained_name=model_config["pretrained_model_path"],
                                                    vocab_size=train_dataset.tokenizer.vocab_size,
                                                    train_context=model_config["train_context"],
                                                    lr=model_config["lr"],
                                                    total_steps=model_config["epochs"] * len(train_dataloader),
                                                    adaptation_part=model_config["adaptation_part"],
                                                    div_factor=model_config["div_factor"],
                                                    end_infill_id=train_dataset.tokenizer.end_infill_id)
    print(text_infill_model)
    text_infill_checkpoint_callback = ModelCheckpoint(filename='best-{epoch}',
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_top_k=1)
    trainer_args = {
        "accelerator": "gpu",
        "log_every_n_steps": 1,
        "max_epochs": model_config["epochs"],
        "default_root_dir": model_config["log_dir"],
        "callbacks": text_infill_checkpoint_callback
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.fit(text_infill_model, train_dataloader, val_dataloader)
    # Plot graphics
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    t_reader.plot_text_infill_tensorboard_graphics()
