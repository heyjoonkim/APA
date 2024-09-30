import pdb

from typing import Dict, List

from datasets import Dataset
import os

from data.dataclass import BaseDataset
from utils.data import load_json


class AmbigFreebaseQA(BaseDataset):
    def __init__(self, 
                 name:str, 
                 data_path:str,
                 seed:int=1234) -> None:
        
        super().__init__(name=name, data_path=data_path, seed=seed)

        

    ## load dataset from file ##    
    def _load_dataset(self, filename:str='test.jsonl') -> List[Dict]:
        assert os.path.isdir(self.data_path), f'{self.data_path} is not a directory.'

        # data_file = os.path.join(self.data_path, f'{self.split}.json')
        data_file = os.path.join(self.data_path, filename)
        assert os.path.isfile(data_file), f'{data_file} is not a file.'

        ## type(dataset) = List[Dict]
        ## Dict.keys()   = is_ambiguous, answers, disambiguations, id, question
        dataset = load_json(path=data_file)

        return dataset


    def get_dataset(self) -> Dataset:
        
        ## load dataset from file
        test_data = self._load_dataset(filename='test.jsonl')

        # TODO: make train and validation sets?

        return dict(
            train=None, 
            validation=None,
            test=test_data,
        )
