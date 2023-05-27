import re
from pathlib import Path
from typing import Iterable, Dict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorboard.backend.event_processing import event_accumulator
from torch import nn
from torch.utils.data import DataLoader


def get_version(path: Path):
    pattern = re.compile(r'version_(\d+)$')
    match = pattern.match(str(path.name))
    if match is not None:
        return int(match.group(1))
    return -1


def get_most_recent_tensorboard_log_and_ckpt(root_path="./"):
    root_path = Path(root_path)
    log_dirs = sorted([(get_version(p), p) for p in root_path.iterdir() if p.is_dir()])
    last_version_path = log_dirs[-1][1]
    checkpoint_dir = last_version_path / Path("checkpoints")
    return last_version_path.__str__(), checkpoint_dir.glob("*.ckpt").__next__().__str__()


def parse_tensorboard(path: str, scalars: Iterable):
    """Returns a dictionary of numpy arrays for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k))["value"].to_numpy() for k in scalars}


def plot_tensorboard_graphics(data: Dict[str, np.ndarray]):
    subplots = [['val_loss', 'train_loss'],
                ['val_acc', 'train_acc']]
    subplots_titles = [("История ошибки", "Ошибка"),
                       ("История точности", "Точность")]
    labels = {'train_loss': 'Train',
              'train_acc': 'Train',
              'val_loss': 'Valid',
              'val_acc': 'Valid'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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


@torch.no_grad()
def print_classification_report(model: nn.Module, loader: DataLoader, device='cuda:0'):
    model.to(device)
    model.eval()
    preds = []
    all_labels = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs).cpu()
        preds.extend(logits.argmax(1))
        all_labels.extend(labels)

    print(classification_report(all_labels, preds))


def get_rgb_cell_map(float_cell_map):
    cmap = plt.cm.get_cmap('Greens')
    max_value = np.max(np.array(float_cell_map))
    rgb_cell_map = [[cmap(cell / max_value) for cell in row] for row in float_cell_map]
    return rgb_cell_map


@torch.no_grad()
def plot_confusion_matrix(model: nn.Module, loader: DataLoader, device='cuda:0') \
        -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.to(device)
    model.eval()
    faults = []
    id2class = {i: q for q, i in loader.dataset.class_to_idx.items()}
    id2class = list(list(zip(*sorted(id2class.items())))[1])
    confusion_matrix = np.zeros((len(id2class), len(id2class)))
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(1)
        faults += [(inputs[j].cpu(), preds[j].item(), labels[j].item()) for j in torch.where(preds != labels)[0]]
        for pred, label in zip(preds, labels):
            confusion_matrix[label][pred] += 1

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(confusion_matrix.tolist(), rowLabels=id2class, colLabels=id2class,
             cellColours=get_rgb_cell_map(confusion_matrix.tolist()), loc='center')
    fig.tight_layout()
    plt.show()
    return faults
