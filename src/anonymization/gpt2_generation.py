from enum import Enum
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from anonymization.base import Anonymization
from datasets.text_infill_dataset import FromListMarkedUpTextInfillDataset, get_ngram_type, MaskNgramType
from datasets.text_infill_tokenization import TargetType
from models.gpt2_model import PretrainedGPT2TextInfilling


class GPT2GenerationAnonymization(Anonymization):
    def __init__(self, model: PretrainedGPT2TextInfilling, path_to_cashed_data='./data/token/cashed_for_gpt2_anon',
                 other_label='O', is_uncased=False, label2type=None, mask_types: Optional[List[Enum]] = None,
                 pretrained_tokenizer: str = None, max_full_ex_len=256, max_only_context_len=192, overlap=32,
                 eq_max_padding=True, gen_model_batch_size=24, device: str = "cuda:0", **kwargs):
        super().__init__(other_label)
        self.model = model
        self.path_to_cashed_data = path_to_cashed_data
        self.is_uncased = is_uncased
        self.get_type = label2type if label2type is not None else get_ngram_type
        self.mask_types = mask_types if mask_types is not None else list(MaskNgramType)
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_full_ex_len = max_full_ex_len
        self.max_only_context_len = max_only_context_len
        self.overlap = overlap
        self.eq_max_padding = eq_max_padding
        self.gen_model_batch_size = gen_model_batch_size
        self.device = device

    def _get_substitutions(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                           source_text_list: List[List[str]]) -> List[List[str]]:
        temp_ids = list(map(str, range(len(source_text_list))))
        dataset = FromListMarkedUpTextInfillDataset(
            self.path_to_cashed_data,
            (temp_ids, source_text_list, general_category_list),
            '', None, self.is_uncased, False,
            self.other_label,  self.get_type, list(self.mask_types),
            self.pretrained_tokenizer, self.max_full_ex_len, self.max_only_context_len, self.overlap,
            self.eq_max_padding, self.device
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.gen_model_batch_size,
            shuffle=False
        )

        predictions = []
        self.model.end_infill_id = dataset.tokenizer.end_infill_id
        self.model.eval()
        last_record_id = ""
        for batch in tqdm(dataloader):
            record_ids, inputs, tts = batch  # B, L
            with torch.no_grad():
                outputs, _ = self.model.inference(inputs, tts)
            answers_starts = torch.nonzero(tts == TargetType.CONTEXT_INFILL_SEP.value)[:, 1].tolist()
            masks_numbers = (tts == TargetType.CONTEXT_SPECIAL.value).sum(dim=-1).tolist()
            for record_id, answers_start, masks_number, pred in zip(record_ids, answers_starts, masks_numbers, outputs):
                record_id = record_id.split(":")[0]
                if record_id != last_record_id:
                    last_record_id = record_id
                    predictions.append([])
                answers = dataset.tokenizer.parse_answers(pred, answers_start + 1, masks_number)
                predictions[-1].extend(answers)
        return predictions
