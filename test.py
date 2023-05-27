from pathlib import Path

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from transformers import set_seed

from datasets.bert_dataset import XMLDataset
from models.bert_model import BertNER
from utils.confusion import print_classification_report, plot_confusion_matrix, print_failed_predictions
from utils.log_reader import TensorBoardReader

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    data_config = yaml.load(open("configs/i2b2_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    test_dataset = XMLDataset(data_config["validate_data_path"],
                              pretrained_tokenizer=data_config["pretrained_tokenizer_path"],
                              max_length=data_config["max_token_number"])
    test_dataloader = DataLoader(test_dataset, shuffle=False,
                                 batch_size=data_config["batch_size"], drop_last=True)
    # Getting path to the last checkpoint
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    path_to_checkpoint = t_reader.get_ckpt_path()
    # Pytorch lightning
    ner_model = BertNER.load_from_checkpoint(path_to_checkpoint)
    trainer_args = {
        "accelerator": "gpu",
        "logger": False
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.test(ner_model, test_dataloader)
    # Print metrics
    print_classification_report(ner_model, test_dataloader)
    fault_ids = plot_confusion_matrix(ner_model, test_dataloader)
    print_failed_predictions(fault_ids, ner_model, test_dataset)
