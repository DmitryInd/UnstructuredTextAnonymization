from typing import List

from torch.utils.data import DataLoader

from anonymization.base import Anonymization
from datasets.text_infill_dataset import FromListMarkedUpTextInfillDataset, get_ngram_type
from datasets.tokenization import OfficialGPT2Tokenizer
from models.gpt2_model import PretrainedGPT2TextInfilling


class GPT2GenerationAnonymization(Anonymization):
    def __init__(self, model: PretrainedGPT2TextInfilling, path_to_cashed_data='./data/token/cashed_for_gpt2_anon',
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
        temp_ids = list(map(str, range(len(source_text_list))))
        dataset = FromListMarkedUpTextInfillDataset(
            self.path_to_cashed_data,
            (temp_ids, source_text_list, general_category_list),
            '', None, self.is_uncased, False,
            self.other_label, get_ngram_type,
            self.pretrained_tokenizer, self.max_sent_len, self.overlap, self.eq_max_padding,
            self.device
        )
        dataloader = DataLoader(
            dataset,
            batch_size=24,
            shuffle=False,
            num_workers=10,
            pin_memory=False,
            persistent_workers=True
        )
        predictions = []
        self.model.eval()
        for batch in dataloader:
            _, inputs, tts = batch  # B, L
            outputs = self.model.inference(inputs, tts)
            for pred in outputs:
                start = list(pred).index(dataset.tokenizer.start_infill_id)
                united_text = self._unite_context_answer(list(outputs[:start]), list(outputs[start+1:]),
                                                         dataset.tokenizer)
                predictions.append(dataset.tokenizer.decode(united_text))

        return predictions

    @staticmethod
    def _unite_context_answer(context: List[int], answer: List[int], tokenizer: OfficialGPT2Tokenizer) -> List[int]:
        i = 0
        while i < len(context):
            if not answer:
                break
            token = context[i]
            if token in tokenizer.mask_type_to_id.values():
                try:
                    end_answer = answer.index(tokenizer.end_infill_id)
                except ValueError:
                    end_answer = len(answer)
                context = context[:i] + answer[:end_answer] + context[i+1:]
                answer = answer[end_answer + 1:]
                i += end_answer
            else:
                i += 1

        return context

