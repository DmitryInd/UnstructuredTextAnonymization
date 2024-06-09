from typing import List, Tuple, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets.ner_tokenization import NERTokenizer
from datasets.ner_dataset import NerDataset


class Statistics:
    def __init__(self, model: nn.Module, loader: DataLoader, device='cuda:0'):
        assert isinstance(loader.dataset, NerDataset), "Dataset in loader must be NerDataset"
        assert loader.dataset.index2label[-1] == loader.dataset.pad_label, "Last label must be padding"

        self.tokenizer: NERTokenizer = loader.dataset.tokenizer
        self.index2label = loader.dataset.index2label
        self.label2index = loader.dataset.label2index
        self.pad_index = self.label2index[loader.dataset.pad_label]
        self.other_index = self.label2index[loader.dataset.other_label]
        self.united_records, self.record_ids, self.true_labels, self.predicted_labels \
            = self._compute_true_and_predicted_labels(model.to(device), loader)
        self.confusion_matrix = self._compute_confusion_matrix()
        self.fault_ids = self._get_fault_record_ids()

    @torch.no_grad()
    def _compute_true_and_predicted_labels(self, model: nn.Module, loader: DataLoader) \
            -> Tuple[Dict[str, Tuple[List[int], List[int], np.ndarray]], List[str], List[int], List[int]]:
        """
        return: {record id: (токены - C, вероятности истинных меток - L x C, логиты предсказанных меток - L x C ), ...},
                объединённый список record id,
        """
        model.eval()
        # Firstly, collect all results for all records
        records: Dict[str, Tuple[List[int], List[List[int]], List[np.ndarray], List[List[int]]]] = dict()
        for batch_ids, inputs, labels in tqdm(loader, desc='Computing predictions'):
            batch_pred = model(inputs.to(model.device)).softmax(dim=-1).cpu().numpy()  # B, L, C
            for sample_id, tokens, prediction, label in zip(batch_ids, inputs.tolist(), batch_pred, labels.tolist()):
                sample_id = sample_id.split(':')
                record_id, offset = ':'.join(sample_id[:-1]), sample_id[-1]
                if record_id not in records:
                    records[record_id] = ([], [], [], [])
                records[record_id][0].append(int(offset))
                records[record_id][1].append(tokens)
                records[record_id][2].append(prediction)
                records[record_id][3].append(label)

        # Secondly, unite all records predictions
        united_records: Dict[str, Tuple[List[int], List[int], np.ndarray]] = dict()
        record_ids = []
        predictions = []
        true_labels = []
        for record_id in tqdm(records.keys(), desc='Uniting records'):
            tokens_set, predictions_set = self.tokenizer.unite_labels(*records[record_id][:-1])
            _, labels_set = self.tokenizer.unite_labels(offsets=records[record_id][0],
                                                        token_segments_list=records[record_id][1],
                                                        label_segments_list=records[record_id][3])
            labels_set = labels_set.argmax(axis=-1).tolist()
            united_records[record_id] = (tokens_set, labels_set, predictions_set)
            record_ids.extend([record_id] * predictions_set.shape[0])
            predictions.extend(predictions_set.argmax(axis=-1).tolist())
            true_labels.extend(labels_set)

        return united_records, record_ids, true_labels, predictions

    def _compute_confusion_matrix(self):
        confusion_matrix = np.zeros((len(self.index2label) - 1, len(self.index2label) - 1))
        for true_label, pred_label in zip(self.true_labels, self.predicted_labels):
            confusion_matrix[true_label][pred_label] += 1

        return confusion_matrix

    def _get_fault_record_ids(self, target_index=-1):
        faults = set()
        for record_id, true_label, pred_label in zip(self.record_ids, self.true_labels, self.predicted_labels):
            if true_label == self.pad_index or (target_index != -1 and true_label != target_index):
                continue
            if true_label != pred_label:
                faults.add(record_id)

        return list(sorted(list(faults)))

    def get_classification_report(self):
        # Exclude 'other' index
        target_names, labels = [], []
        for i, name in enumerate(self.index2label[:-1]):
            if i != self.other_index:
                target_names.append(name)
                labels.append(i)
        return classification_report(self.true_labels,
                                     self.predicted_labels,
                                     labels=labels,
                                     target_names=target_names,
                                     digits=4,
                                     zero_division=0)

    @staticmethod
    def _get_rgb_cell_map(float_cell_map: np.ndarray):
        assert len(float_cell_map.shape) == 2, "Confusion matrix has more than two dimensions"
        cmap = plt.cm.get_cmap('Wistia')
        max_values = np.sum(float_cell_map, axis=1)
        rgb_cell_map = [[cmap(cell / (max_value + 1e-5)) for cell in row]
                        for row, max_value in zip(float_cell_map, max_values)]
        return rgb_cell_map

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix
        """
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title('y: true/x: predicted labels')
        ax.table(self.confusion_matrix.tolist(), rowLabels=self.index2label[:-1], colLabels=self.index2label[:-1],
                 cellColours=self._get_rgb_cell_map(self.confusion_matrix), loc='center')
        fig.tight_layout()
        plt.show()

    @torch.no_grad()
    def print_random_failed_predictions(self, num_samples: int = 5, fault_record_ids: List[int] = None):
        if fault_record_ids is None:
            fault_record_ids = self.fault_ids
        num_samples = min(num_samples, len(fault_record_ids))
        internal_ids = np.random.choice(np.arange(len(fault_record_ids)), num_samples, replace=False)
        print('Wrongly predicted examples:')
        for i in internal_ids:
            record_id = fault_record_ids[i]
            token_ids, label_ids, prediction = self.united_records[record_id]
            true_segments, true_label_ids = self.tokenizer.decode(token_ids, label_ids)
            pred_segments, pred_label_ids = self.tokenizer.decode(token_ids, prediction)
            print('_' * 5 + f' Record {record_id} ' + '_' * 5)
            print(tabulate([['Sentence:'] + true_segments,
                            ['True labels:'] + [self.index2label[index] for index in true_label_ids]],
                           tablefmt='orgtbl'))
            print(tabulate([['Sentence:'] + pred_segments,
                            ['Pred labels:'] + [self.index2label[index] for index in pred_label_ids]],
                           tablefmt='orgtbl'))

    def get_specific_failed_predictions(self, target_type: str, num_samples: int = 2):
        specific_faults = self._get_fault_record_ids(self.label2index[target_type])
        self.print_random_failed_predictions(num_samples, specific_faults)
        return specific_faults
