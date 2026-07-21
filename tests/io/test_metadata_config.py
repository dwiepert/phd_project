"""
Test Metadata Configuration

Author(s): Dani Wiepert
Last modified: 06/2026
"""
#IMPORTS
##built-in 
from pathlib import Path

##third-party 
import pytest
import pandas as pd

##local
from acm.io import MetadataConfig

def create_metadata(path, excel=True):
    data = {'uid': ['000', '001', '002', '003', '004','005', '006'], 
            'speakerID': ['aaa', 'bbb', 'ccc', 'ddd', 'eee','fff', 'ggg'], 
            'file_name': ['harvard', 'harvard', 'harvard', 'harvard', 'harvard', 'harvard', 'random'], 
            'task': ['sentence-repetition','sentence-repetition', 'sentence-repetition', 'sentence-repetition', 'sentence-repetition', 'random', 'sentence-repetition'], 
            'split': ['train', 'val','val', 'test', 'random', 'train', 'test'],
            'date': ['01-01-2026','01-01-2026','01-01-2026','01-01-2026','01-01-2026','01-01-2026','01-01-2026'], 
            'session': [1,1,1,1,1,1,1]}
    
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{path}.csv', index=False)
    if excel:
        data_df.to_excel(f'{path}.xlsx')
    return data_df

def create_metadata2():
    data = {'uid': ['000', '001', '002'], 
            'speakerID': ['aaa', 'bbb', 'ccc'], 
            'file_name': ['harvard', 'harvard', 'harvard'], 
            'task': ['sentence-repetition','sentence-repetition', 'sentence-repetition'], 
            'split': ['train', 'val','val'],
            'date': ['01-01-2026','01-01-2026','01-01-2026'], 
            'session': [1,1,1]}
    
    data_df = pd.DataFrame(data)
    return data_df

def test_init(data_dir):
    #str/path metadata
    metadata = create_metadata(str(data_dir / 'metadata'), False)
    metadata_path = data_dir / 'metadata.csv'

    #test metadata not string/path/dataframe
    with pytest.raises(AssertionError):
        m = MetadataConfig(1000, uid_col='uid')
    
    # test str 
    m1 = MetadataConfig(metadata=metadata_path, uid_col="uid")
    # test path
    m2 = MetadataConfig(metadata=Path(metadata_path), uid_col="uid")
    # test preloaded
    m3 = MetadataConfig(metadata=metadata, uid_col="uid")

    #test invalid columns
    #uid
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='random')
    #speaker
    m = MetadataConfig(metadata,uid_col='uid', speaker_col='speakerID')
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid', speaker_col='random')
    #file name
    m = MetadataConfig(metadata,uid_col='uid', file_col='file_name')
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid', file_col='random')
    #task
    m = MetadataConfig(metadata,uid_col='uid', task_col='task')
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid', task_col='random')
    #task filtering
    m_task = MetadataConfig(metadata, uid_col='uid', task_col='task', tasks=['sentence-repetition'])
    assert len(m_task.metadata) == 6, 'Filtering should make length 6'
    #task not present in column
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid', task_col='split',tasks=['sentence-repetition'])
    #session
    m = MetadataConfig(metadata,uid_col='uid', session_col='session')
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid',  session_col='random')
    #date
    m = MetadataConfig(metadata, uid_col='uid', date_col='date')
    with pytest.raises(AssertionError):
        m = MetadataConfig(metadata,uid_col='uid', date_col='random')
    #date not compatible w datetime
    with pytest.raises(pd._libs.tslibs.parsing.DateParseError):
        m = MetadataConfig(metadata, uid_col='uid', date_col='split')

def test_update(data_dir):
    #str/path metadata
    metadata = create_metadata(str(data_dir / 'metadata'), False)

    metadata2 = create_metadata2()
    m = MetadataConfig(metadata=metadata, uid_col='uid')
    m.update_metadata(metadata2)

    metadata2 = metadata2.set_index("uid")
    assert m.metadata.equals(metadata2)

#TODO: test filtering by diagnoses and filtering by tasks