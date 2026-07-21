
#IMPORTS
##built-in
from pathlib import Path
import random

##third-party
import pandas as pd
import pytest

from acm.io import AudioConfig, MetadataConfig, GCSConfig
from acm.dataset import AudioDataset

def create_metadata(path, excel=True):
    data = {'uid': ['000', '001', '002', '003', '004','005', '006'], 
            'speakerID': ['aaa', 'aaa', 'ccc', 'ccc', 'eee','fff', 'ggg'], 
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

def test_init(data_dir):
    metadata = create_metadata(str(data_dir / 'metadata'), False)
    d = AudioDataset(AudioConfig(data_dir), MetadataConfig(metadata, uid_col="uid", file_col='file_name'), dataset_name="basic")

    md = d.get_metadata()
    assert isinstance(md, pd.DataFrame)
    assert len(d) == 7 #-1 for filtering

def test_get_item(data_dir):
    metadata = create_metadata(str(data_dir / 'metadata'), False)
    d0 = AudioDataset(AudioConfig(data_dir), MetadataConfig(metadata,uid_col='uid', file_col='file_name', task_col='task', tasks=['sentence-repetition']), dataset_name="basic")

    #invalid index
    with pytest.raises(IndexError):
       sample0 = d0[6]

    #non-existent audio 
    with pytest.raises(RuntimeError):
        sample0 = d0[len(d0)-1]

    #audio exists, no truncate
    sample1 = d0[0]
    assert sample1['sample_rate'] == 16000

    #audio exists, truncate
    d1 = AudioDataset(AudioConfig(data_dir, truncate=.5), MetadataConfig(metadata,uid_col="uid", file_col='file_name', task_col="task", tasks=['sentence-repetition']), dataset_name="basic")
    sample2 = d1[0]
    assert sample2['sample_rate'] == 16000
    assert sample1['waveform'].shape[1] > sample2['waveform'].shape[1]

def test_split_by_file(data_dir):
    metadata = create_metadata(str(data_dir / 'metadata'), False)
    d0 = AudioDataset(AudioConfig(data_dir), MetadataConfig(metadata,uid_col='uid', file_col='file_name', task_col='task', tasks=['sentence-repetition']), dataset_name="basic")

    #invalid train/val/test combinations 
    tr,v,te = AudioDataset.split_data_by_file(d0, train=0.5, val=0.34)
    assert len(tr) == 3
    assert len(v) == 2
    assert len(te) == 1
    assert not tr.get_metadata().index.isin(v.get_metadata().index).any()
    assert not v.get_metadata().index.isin(te.get_metadata().index).any()
    assert not tr.get_metadata().index.isin(te.get_metadata().index).any()

    tr,v,te = AudioDataset.split_data_by_file(d0, train=0.9, val=0)
    assert len(tr) == 5
    assert v is None
    assert len(te) == 1
    assert not tr.get_metadata().index.isin(te.get_metadata().index).any()

    tr,v,te = AudioDataset.split_data_by_file(d0, train=0, val=0.34)
    assert tr is None
    assert len(v) == 2
    assert len(te) == 4
    assert not v.get_metadata().index.isin(te.get_metadata().index).any()

    tr,v,te = AudioDataset.split_data_by_file(d0, train=0, val=0)
    assert tr is None
    assert v is None
    assert len(te) == 6
    tr,v,te = AudioDataset.split_data_by_file(d0, train=1, val=0)
    assert len(tr) == 6
    assert v is None
    assert te is None
    tr,v,te = AudioDataset.split_data_by_file(d0, train=0, val=1)
    assert tr is None
    assert len(v) == 6
    assert te is None
    
    with pytest.raises(AssertionError):
        AudioDataset.split_data_by_file(d0, train=0.9, val=0.15)

def test_split_by_speaker(data_dir):
    metadata = create_metadata(str(data_dir / 'metadata'), False)
    d0 = AudioDataset(AudioConfig(data_dir), MetadataConfig(metadata,uid_col='uid', file_col='file_name', speaker_col='speakerID',task_col='task', tasks=['sentence-repetition']), dataset_name="basic")

    #invalid train/val/test combinations 
    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=0.5, val=0.25, seed=42)
    #assert len(tr) == 3
    #assert len(v) == 1
    #assert len(te) == 2
    assert not tr.get_metadata().index.isin(v.get_metadata().index).any()
    assert not v.get_metadata().index.isin(te.get_metadata().index).any()
    assert not tr.get_metadata().index.isin(te.get_metadata().index).any()
    train_speakers = list(tr.get_metadata()['speakerID'])
    val_speakers = list(v.get_metadata()['speakerID'])
    test_speakers = list(te.get_metadata()['speakerID'])
    assert set(train_speakers).isdisjoint(val_speakers)
    assert set(train_speakers).isdisjoint(test_speakers)
    assert set(val_speakers).isdisjoint(test_speakers)

    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=0.9, val=0, seed=42)
    #assert len(tr) == 5
    assert v is None
    #assert len(te) == 1
    #assert not tr.get_metadata().index.isin(te.get_metadata().index).any()
    train_speakers = list(tr.get_metadata()['speakerID'])
    test_speakers = list(te.get_metadata()['speakerID'])
    assert set(train_speakers).isdisjoint(test_speakers)
   

    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=0, val=0.25, seed=42)
    assert tr is None
    #assert len(v) == 1
    #assert len(te) == 5
    assert not v.get_metadata().index.isin(te.get_metadata().index).any()
    val_speakers = list(v.get_metadata()['speakerID'])
    test_speakers = list(te.get_metadata()['speakerID'])
    assert set(val_speakers).isdisjoint(test_speakers)

    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=0, val=0, seed=42)
    assert tr is None
    assert v is None
    assert len(te) == 6
    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=1, val=0, seed=42)
    assert len(tr) == 6
    assert v is None
    assert te is None
    tr,v,te = AudioDataset.split_data_by_speaker(d0, train=0, val=1, seed=42)
    assert tr is None
    assert len(v) == 6
    assert te is None
    
    with pytest.raises(AssertionError):
        AudioDataset.split_data_by_file(d0, train=0.9, val=0.15, seed=42)

#TODO: redo most of this