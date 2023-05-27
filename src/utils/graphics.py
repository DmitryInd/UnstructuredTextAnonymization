from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader


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
