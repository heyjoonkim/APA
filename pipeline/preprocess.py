import copy
import logging
from typing import Dict, Sequence, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from pipeline.templates import INFERENCE_TEMPLATES

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100

# FROM : https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L157
@dataclass
class DataCollatorForSupervisedDataset(object):

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# FROM : https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L88
def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            # padding="longest",
            # max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# FROM : https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L112
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    
    examples = [s + t for s, t in zip(sources, targets)]
    
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# FROM : https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L127
class SupervisedDataset(Dataset):

    def __init__(self, dataset:List[Dict], tokenizer: PreTrainedTokenizer, template_id: int) -> None:
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        template = INFERENCE_TEMPLATES[template_id]

        sources = [template.format(example.get('question')) for example in dataset]

        targets = [f"{example['prediction']}{tokenizer.eos_token}" for example in dataset]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



def make_supervised_data_module(tokenizer:PreTrainedTokenizer, dataset:List[Dict], template_id:int):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=dataset, template_id=template_id)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
