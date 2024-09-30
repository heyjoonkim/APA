



class BaseDataset:

    def __init__(
                self,
                name:str, 
                data_path:str,
                seed:int=1234) -> None:
        
        self.name = name
        self.data_path = data_path
        self.seed = seed

    
    def _load_dataset(self):
        raise NotImplementedError('This method should be implemented in the child class.')
    
    def _get_dataset(self):
        raise NotImplementedError('This method should be implemented in the child class.')
    