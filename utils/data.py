
import os

import json
import pickle

# check if path exists, if not, create it 
def _check_path(path:str):
    directory = os.path.dirname(path)

    if os.path.isdir(directory) is False:
        os.makedirs(directory, exist_ok=True)


##################
#                #
#     Pickle     #
#                #
##################

def load_pkl(path:str):

    if not os.path.exists(path):
        return None

    with open(path, 'rb') as infile:
        pkl_file = pickle.load(infile)
    
    return pkl_file

def save_pkl(data, path):
    _check_path(path)
    
    with open(path, 'wb') as outfile:
        pickle.dump(data, outfile)


################
#              #
#     JSON     #
#              #
################
        
def load_json(path:str):
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as infile:
        data = json.load(infile)
    return data

def save_json(data, path):
    _check_path(path)
    
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile)