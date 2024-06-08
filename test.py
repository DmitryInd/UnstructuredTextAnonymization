import sys
from pathlib import Path

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(1, "./src")
from datasets.ner_dataset import get_ner_dataset
from models.bert_model import PretrainedBertNER
from utils.ner_metrics import Statistics
from utils.log_reader import TensorBoardReader
from anonymization.ref_book import ReferenceBookAnonymization

if __name__ == '__main__':
    set_seed(42)
    # Config initialisation
    anon_config = yaml.load(open("configs/ref_book_anonymization_config.yaml", 'r'), Loader=yaml.Loader)
    data_config = yaml.load(open("configs/i2b2-2014_data_config.yaml", 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open("configs/bert-large_model_config.yaml", 'r'), Loader=yaml.Loader)
    # Data processing
    anonymization = ReferenceBookAnonymization(**anon_config,
                                               other_label=data_config['other_label'])
    test_dataset = get_ner_dataset(path_to_folder=data_config["validate_data_path"],
                                   anonymization=anonymization, device='cpu', **data_config)
    test_dataloader = DataLoader(test_dataset, shuffle=False,
                                 batch_size=data_config["batch_size"],
                                 collate_fn=test_dataset.get_collate_fn(),
                                 num_workers=10,
                                 pin_memory=False,
                                 persistent_workers=True)
    # Getting path to the last checkpoint
    t_reader = TensorBoardReader(Path(model_config["log_dir"]) / Path("lightning_logs"))
    path_to_checkpoint = t_reader.get_ckpt_path()
    # Pytorch lightning
    ner_model = PretrainedBertNER.load_from_checkpoint(path_to_checkpoint, strict=False)
    print(ner_model)
    trainer_args = {
        "accelerator": "gpu",
        "logger": False
    }
    trainer = pl.Trainer(**trainer_args, enable_progress_bar=True)
    trainer.test(ner_model, test_dataloader)
    # Print metrics
    stats = Statistics(ner_model, test_dataloader)
    print(stats.get_classification_report())
    stats.plot_confusion_matrix()
    stats.print_random_failed_predictions()
