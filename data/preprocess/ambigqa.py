
import pdb
import os
import logging
from time import time
from typing import List, Dict

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from utils import seed_everything
from utils.logging import logger_init
from utils.data import load_json, save_json


logger = logging.getLogger(__name__)


def deduplicate_annotations(annotations:List) -> Dict:
    SINGLE = 'singleAnswer'
    MULTIPLE = 'multipleQAs'

    annotation_types = [annotation.get('type') for annotation in annotations]
    is_ambiguous = True if MULTIPLE in annotation_types else False


    if is_ambiguous:
        answers = None
        disambiguations = list()
        for annotation in annotations:
            annotation_type = annotation.get('type')
            if annotation_type == MULTIPLE:
                # type          : List[Dict]
                # Dict.keys()   : question, answer
                qa_pairs = annotation.get('qaPairs')
                disambiguations += qa_pairs
    else:
        answers = list()
        disambiguations = None
        for annotation in annotations:
            answer = annotation.get('answer')
            answers += answer
        # remove duplicate samples
        answers = list(set(answers))

    output = dict(
        is_ambiguous=is_ambiguous,          # bool
        answers=answers,                    # List[str]
        disambiguations=disambiguations,    # List[Dict], Dict.keys(): question:str, answer:List
    )

    return output


@hydra.main(config_path='../../configs/data/', config_name='ambigqa.yaml')
def main(args: DictConfig) -> None:

    seed = args.seed
    seed_everything(seed)
    
    data_path = args.path.data
    assert os.path.isdir(data_path), f'Data path does not exist: {data_path}'

    output_path = args.path.output
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    logger_path = args.path.log
    logger_init(logger=logger, output_dir=logger_path, save_as_file=True)


    ##################
    #                #
    # LOAD TRAIN SET #
    #                #
    ##################

    splits = ['train', 'dev', 'test']
    for split in splits:

        # filename = os.path.join(data_path, f'{split}_light.json')
        filename = os.path.join(data_path, f'{split}_full.json')
        assert os.path.isfile(filename), f'Data file does not exist: {filename}'
        output_filename = os.path.join(output_path, f'{split}.json')

        samples = load_json(path=filename)
        logger.info(f'# {split} set: {len(samples)}')


        revised_samples = list()
        for sample in tqdm(samples, total=len(samples), desc='Preprocessing Train Set'):
            sample_id = sample.get('id')
            question = sample.get('question')
            annotations = sample.get('annotations')

            # dict.keys(): is_ambiguous:bool, answers:List[str], disambiguations:List[Dict]
            output_dict = deduplicate_annotations(annotations=annotations)

            output_dict.update(
                dict(
                    id=sample_id,
                    question=question,
                )
            )
            revised_samples.append(output_dict)

        save_json(data=revised_samples, path=output_filename)


    logger.info('Done.')



if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    logger.info(f'Total run time : {end_time - start_time} sec.')