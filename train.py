from models.bert_model import BertNER
from datasets.bert_dataset import XMLDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import set_seed
import pytorch_lightning as pl
from utils.log_reader import TensorBoardReader
from pathlib import Path
import yaml

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    data_config = yaml.load(open("configs/i2b2_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    train_dataset = XMLDataset(data_config["train_data_path"],
                               is_uncased=data_config["is_uncased"],
                               pretrained_tokenizer=data_config["pretrained_tokenizer_path"],
                               max_length=data_config["max_token_number"])
    val_dataset = XMLDataset(data_config["validate_data_path"],
                             is_uncased=data_config["is_uncased"],
                             pretrained_tokenizer=data_config["pretrained_tokenizer_path"],
                             max_length=data_config["max_token_number"])
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=data_config["batch_size"])
    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                batch_size=data_config["batch_size"], drop_last=True)
    # Pytorch lightning
    ner_model = BertNER(pretrained_name=model_config["pretrained_model_path"],
                        encoder_vocab_size=len(train_dataset.tokenizer.index2word),
                        num_classes=len(train_dataset.index2label),
                        lr=model_config["lr"],
                        total_steps=model_config["epochs"] * len(train_dataloader),
                        div_factor=model_config["div_factor"],
                        other_index=train_dataset.label2index[data_config["other_label"]])
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
