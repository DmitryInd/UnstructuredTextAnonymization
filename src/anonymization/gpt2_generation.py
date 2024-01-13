from typing import List

from anonymization.base import Anonymization
from datasets.text_infill_dataset import FromListMarkedUpTextInfillDataset, get_ngram_type


class GPT2GenerationAnonymization(Anonymization):
    def __init__(self, model, path_to_cashed_data='./data/token/cashed_for_gpt2_anon',
                 other_label='O', is_uncased=False,
                 pretrained_tokenizer: str = None, max_sent_len=768, overlap=True, eq_max_padding=True,
                 device: str = None):
        super().__init__()
        self.model = model
        self.path_to_cashed_data = path_to_cashed_data
        self.other_label = other_label
        self.is_uncased = is_uncased
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_sent_len = max_sent_len
        self.overlap = overlap
        self.eq_max_padding = eq_max_padding
        self.device = device

    def __call__(self, general_category_list: List[List[str]], specific_category_list: List[List[str]],
                 source_text_list: List[List[str]]) -> List[List[str]]:
        temp_ids = list(range(len(source_text_list)))
        dataset = FromListMarkedUpTextInfillDataset(
            self.path_to_cashed_data,
            (temp_ids, source_text_list, general_category_list),
            '', None, self.is_uncased, False,
            self.other_label, get_ngram_type,
            self.pretrained_tokenizer, self.max_sent_len, self.overlap, self.eq_max_padding,
            self.device
        )
        # TODO
