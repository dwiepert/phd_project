"""
Test CustomDataset

Author(s): Dani Wiepert
Last modified: 01/2026
"""

#IMPORTS
##built-in
import os
from pathlib import Path

##third-party
import pandas as pd
import pytest

from acm.io import CustomDataset

def create_metadata(path, excel=True):
    data = {'uid': ['000', '001', '003', '004', '005','006'], 
            'speakerID': ['aaa', 'bbb', 'ccc', 'ddd', 'eee','fff'], 
            'file_name': ['harvard', 'harvard', 'harvard', 'harvard', 'harvard', 'harvard'], 
            'task': ['sentence-repetition','sentence-repetition', 'sentence-repetition', 'sentence-repetition', 'sentence-repetition', 'random'], 
            'split': ['train', 'val','val', 'test', 'random', 'train']}
    
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{path}.csv', index=False)
    if excel:
        data_df.to_excel(f'{path}.xlsx')

### TESTS ###
def test_path_names():
    str_audio_dir = 'tests/data'
    path_audio_dir = Path(str_audio_dir)
    base_path = 'tests/data/metadata'
    str_metadata_path = f'{base_path}.csv'
    alt_metadata_path_0 = f'{base_path}_1.csv'
    alt_metadata_path_1 = f'{base_path}.xlsx'
    path_metadata_path = Path(str_metadata_path)
    create_metadata(base_path)

    #path doesn't exist audio_directory
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir+'0', metadata_path=str_metadata_path)

    #exists but different extension
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, audio_ext='.flac')

    #exists and proper extension
    d0 = CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path)
    d1 = CustomDataset(audio_dir=path_audio_dir, metadata_path=str_metadata_path)

    #path doesn't exist metadata_path
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=alt_metadata_path_0)

    #exists but different extension
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=alt_metadata_path_1)

    #exists and proper extension
    d2 = CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path)
    d3 =  CustomDataset(audio_dir=str_audio_dir, metadata_path=path_metadata_path)

    os.remove(str_metadata_path)
    os.remove(alt_metadata_path_1)

def test_metadata():
    str_audio_dir = 'tests/data'
    base_path = 'tests/data/metadata'
    str_metadata_path = f'{base_path}.csv'
    create_metadata(base_path, False)

    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, uid_col='random')
    
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, audio_col='random')

    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, speaker_col='random')
        
    with pytest.raises(AssertionError):
        CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, task_col='random')
    
    d0 = CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path)
    md = d0.get_metadata()
    assert  all(md[d0.task_col].isin(d0.tasks))
    
    new_metadata = ['metadata']
    with pytest.raises(AssertionError):
        d0.update_metadata(new_metadata)
    new_metadata = md[md['split'] == 'val']
    d0.update_metadata(new_metadata)

    #TODO: try updating metadata with false files
    md0 = md.copy()
    md0 = md0.drop(columns=['task'])
    with pytest.raises(AssertionError):
        d0.update_metadata(md0)
    
    md1 = md.copy()
    md1.reset_index(drop=True)
    md1.index.name ='random'
    with pytest.raises(AssertionError):
        d0.update_metadata(md1)

    md2 = md.copy()
    md2 = md2.drop(columns=['speakerID'])
    with pytest.raises(AssertionError):
        d0.update_metadata(md2)
    
    md3 = md.copy()
    md3 = md0.drop(columns=['file_name'])
    with pytest.raises(AssertionError):
        d0.update_metadata(md3)

    d1 = CustomDataset(audio_dir=str_audio_dir, metadata_path=str_metadata_path, tasks=['random'])
    new_md4 = md[md['task'] != 'random']
    d1.update_metadata(new_md4)
    os.remove(str_metadata_path)
    

def test_get_item():
    #TODO: TEST GET ITEM (RESAMPLE+MONOPHONIC)
    #TODO: TEST GET ITEM (+TRUNCATE)
    return

def test_split():
    #TODO: TEST SPLIT
    return