import re
from pathlib import Path
from typing import Iterable, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


class TensorBoardReader:
    def __init__(self, root_path="./"):
        self.default_metrics_list = ['train_loss', 'train_recall', 'train_precision', 'train_f1',
                                     'val_loss', 'val_recall', 'val_precision', 'val_f1']
        self.root_path = Path(root_path)
        log_dirs = sorted([(self._get_version(p), p) for p in self.root_path.iterdir() if p.is_dir()])
        self.last_log_path = log_dirs[-1][1].__str__()
        self.version_paths = {version: str(path_to_dir) for version, path_to_dir in log_dirs}
        self.last_ckpt_path = self._get_ckpt_path(self.last_log_path)

    @staticmethod
    def _get_version(path: Path):
        pattern = re.compile(r'version_(\d+)$')
        match = pattern.match(str(path.name))
        if match is not None:
            return int(match.group(1))
        return -1

    @staticmethod
    def _get_ckpt_path(log_dir_path: str):
        checkpoint_dir = Path(log_dir_path) / Path("checkpoints")
        return checkpoint_dir.glob("*.ckpt").__next__().__str__()

    def get_ckpt_path(self, version: int = -1):
        """
        Return path to file with weight in checkpoint from the specified launch version.
        If there are several checkpoints, the function proceeds only the first one.
        """
        return self._get_ckpt_path(self.version_paths[version] if version != -1 else self.last_log_path)

    def get_scalars(self, version: int = -1, scalars_names: Iterable = None) -> Dict[str, np.ndarray]:
        """Returns a dictionary of numpy arrays for each requested scalar"""
        if scalars_names is None:
            scalars_names = self.default_metrics_list
        ea = event_accumulator.EventAccumulator(
            self.version_paths[version] if version != -1 else self.last_log_path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        _absorb_print = ea.Reload()
        # make sure the scalars are in the event accumulator tags
        assert all(
            s in ea.Tags()["scalars"] for s in scalars_names
        ), "some scalars were not found in the event accumulator"
        return {k: pd.DataFrame(ea.Scalars(k))["value"].to_numpy() for k in scalars_names}

    def plot_tensorboard_graphics(self, version: int = -1):
        data = self.get_scalars(version)
        subplots = [['val_loss', 'train_loss'],
                    ['val_recall', 'train_recall'],
                    ['val_precision', 'train_precision'],
                    ['val_f1', 'train_f1']]
        subplots_titles = [("История ошибки", "Ошибка"),
                           ("История Recall", "Recall"),
                           ("История Precision", "Precision"),
                           ("История $F_1$ score", "$F_1$ score")]
        labels = {'train_loss': 'Train',
                  'train_recall': 'Train',
                  'train_precision': 'Train',
                  'train_f1': 'Train',
                  'val_loss': 'Valid',
                  'val_recall': 'Valid',
                  'val_precision': 'Valid',
                  'val_f1': 'Valid'}

        fig, axes = plt.subplots(1, len(subplots), figsize=(7*len(subplots), 5))
        for i, (scalar_names, (title, y_label)) in enumerate(zip(subplots, subplots_titles)):
            for name in scalar_names:
                axes[i].plot(list(range(1, 1 + len(data[name]))),
                             data[name], label=labels[name])

            axes[i].set_title(title)
            axes[i].set_xlabel("Эпоха")
            axes[i].set_ylabel(y_label)
            axes[i].grid(ls=':')
            axes[i].legend()
        plt.show()
