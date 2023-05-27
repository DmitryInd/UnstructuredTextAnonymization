from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from datasets.bert_dataset import XMLDataset
from tabulate import tabulate


@torch.no_grad()
def print_classification_report(model: nn.Module, loader: DataLoader, device='cuda:0'):
    model.to(device)
    model.eval()
    predictions = []
    all_labels = []
    for _, inputs, labels in loader:
        inputs = inputs.to(device)
        batch_pred = model(inputs).cpu().argmax(2)
        for prediction, label in zip(batch_pred, labels.cpu()):
            predictions.extend(prediction.tolist())
            all_labels.extend(label.tolist())

    print(classification_report(all_labels, predictions))


def get_rgb_cell_map(float_cell_map):
    cmap = plt.cm.get_cmap('Greens')
    max_value = np.max(np.array(float_cell_map))
    rgb_cell_map = [[cmap(cell / max_value) for cell in row] for row in float_cell_map]
    return rgb_cell_map


@torch.no_grad()
def plot_confusion_matrix(model: nn.Module, loader: DataLoader, device='cuda:0') \
        -> List[int]:
    """
    Plot the confusion matrix and return the list of ids of wrong recognized records
    """
    model.to(device)
    model.eval()
    faults = []
    id2class = {i: q for q, i in loader.dataset.label2index.items()}
    id2class = list(list(zip(*sorted(id2class.items())))[1])
    confusion_matrix = np.zeros((len(id2class), len(id2class)))
    for batch_ids, inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs).argmax(2)
        for sample_id, prediction, label in zip(batch_ids, predictions, labels):
            is_mistake = False
            for token_pred, token_label in zip(prediction, label):
                confusion_matrix[token_label][token_pred] += 1
                if token_pred != token_label:
                    is_mistake = True

            if is_mistake:
                faults.append(sample_id)

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(confusion_matrix.tolist(), rowLabels=id2class, colLabels=id2class,
             cellColours=get_rgb_cell_map(confusion_matrix.tolist()), loc='center')
    fig.tight_layout()
    plt.show()
    return faults


def print_failed_predictions(failed_prediction_ids: List[int], model, dataset: XMLDataset, num_samples: int = 5):
    internal_ids = np.random.choice(np.arange(0, len(failed_prediction_ids)), num_samples, replace=False)
    for i in internal_ids:
        record_id = failed_prediction_ids[i].item()
        idx = dataset.record2idx[record_id]
        _, token_ids, label_ids = dataset[idx]
        pred_ids = model(token_ids.unsqueeze(0))[0].argmax(1)
        token_ids, label_ids, pred_ids = token_ids.tolist(), label_ids.tolist(), pred_ids.tolist()
        sentence, true_label_ids = dataset.tokenizer.decode(token_ids, label_ids)
        _, pred_label_ids = dataset.tokenizer.decode(token_ids, pred_ids)
        print('Wrongly predicted examples:')
        print('_'*5 + f' Record {record_id} ' + '_'*5)
        print(tabulate([['Sentence:'] + sentence,
                        ['True labels:'] + [dataset.index2label[index] for index in true_label_ids],
                        ['Pred labels:'] + [dataset.index2label[index] for index in pred_label_ids]],
                       tablefmt='orgtbl'))

