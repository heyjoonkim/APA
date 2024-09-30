
from data.dataclass.ambigqa import AmbigQA
from data.dataclass.ambig_trivia_qa import AmbigTriviaQA
from data.dataclass.situated_qa_geo import SituatedQAGeo
from data.dataclass.situated_qa_temp import SituatedQATemp
from data.dataclass.ambig_web_questions import AmbigWebQuestions
from data.dataclass.ambig_freebase_qa import AmbigFreebaseQA

def get_dataset_class(name:str, data_path:str, seed:int=1234):
        if name == 'ambigqa':
            data_class = AmbigQA(name=name, data_path=data_path, seed=seed)
        elif 'ambig_trivia_qa' in name:
            data_class = AmbigTriviaQA(name=name, data_path=data_path, seed=seed)
        elif 'ambig_wiki_qa' in name:
            data_class = SituatedQAGeo(name=name, data_path=data_path, seed=seed)
        elif 'situated_qa_temp' in name:
            data_class = SituatedQATemp(name=name, data_path=data_path, seed=seed)
        elif 'ambig_web_questions' in name:
            data_class = AmbigWebQuestions(name=name, data_path=data_path, seed=seed)
        elif 'ambig_freebase_qa' in name:
            data_class = AmbigFreebaseQA(name=name, data_path=data_path, seed=seed)
        else:
            raise NotImplementedError(f'Dataset {name} not implemented.')
        
        return data_class
