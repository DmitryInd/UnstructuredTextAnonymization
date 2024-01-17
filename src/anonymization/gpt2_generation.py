from typing import List

from torch.utils.data import DataLoader

from anonymization.base import Anonymization
from datasets.text_infill_dataset import FromListMarkedUpTextInfillDataset, get_ngram_type
from datasets.tokenization import OfficialGPT2Tokenizer, TargetType
from models.gpt2_model import PretrainedGPT2TextInfilling
from tqdm import tqdm


class GPT2GenerationAnonymization(Anonymization):
    def __init__(self, model: PretrainedGPT2TextInfilling, path_to_cashed_data='./data/token/cashed_for_gpt2_anon',
                 other_label='O', is_uncased=False,
                 pretrained_tokenizer: str = None, max_sent_len=192, padding_len=256, overlap=32, eq_max_padding=True,
                 device: str = "cuda:0"):
        super().__init__()
        self.model = model
        self.path_to_cashed_data = path_to_cashed_data
        self.other_label = other_label
        self.is_uncased = is_uncased
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_sent_len = max_sent_len
        self.padding_len = padding_len
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
            self.pretrained_tokenizer, self.max_sent_len, self.overlap, self.eq_max_padding, self.padding_len,
            self.device
        )
        dataloader = DataLoader(
            dataset,
            batch_size=24,
            shuffle=False
        )
        predictions = []
        self.model.end_infill_id = dataset.tokenizer.end_infill_id
        self.model.eval()
        last_record_id = ""
        for batch in tqdm(dataloader):
            record_ids, inputs, tts = batch  # B, L
            outputs = self.model.inference(inputs, tts)
            for record_id, labels, pred in zip(record_ids, tts, outputs):
                record_id = record_id.split(":", 1)[0]
                if record_id != last_record_id:
                    last_record_id = record_id
                    predictions.append([])
                masks_number = (labels == TargetType.CONTEXT_SPECIAL.value).sum().item()
                start = list(pred).index(dataset.tokenizer.start_infill_id)
                answers = self._parse_answers(pred[start+1:].tolist(), dataset.tokenizer)
                for i in range(masks_number):
                    answer = answers[i] if i < len(answers) else []
                    predictions[-1].append(dataset.tokenizer.decode(answer))

        new_texts = []
        for source_text, categories, answers in zip(source_text_list, general_category_list, predictions):
            new_texts.append([])
            i = 0
            for text, category in zip(source_text, categories):
                if category != self.other_label:
                    if i < len(answers):
                        new_texts[-1].append(answers[i])
                        i += 1
                    else:
                        new_texts[-1].append("")
                else:
                    new_texts[-1].append(text)

        return new_texts

    @staticmethod
    def _parse_answers(answers: List[int], tokenizer: OfficialGPT2Tokenizer) -> List[List[int]]:
        answers_list = []
        while answers:
            try:
                end_answer = answers.index(tokenizer.end_infill_id)
            except ValueError:
                end_answer = len(answers)
            answers_list.append(answers[:end_answer])
            answers = answers[end_answer + 1:]

        return answers_list

