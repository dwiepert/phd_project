"""
Base Dataset implementation 

Author(s): Dani Wiepert 
Last modified: 06/2027
"""

#IMPORTS
#built-in
from typing import List, Union, Tuple
import random
from pathlib import Path

#third-party
import pandas as pd
import torch
from torch.utils.data import Dataset

#local
from acm.io._audio_config import AudioConfig
from acm.io._metadata_config import MetadataConfig

class AudioDataset(Dataset):
    '''
    Audio Dataset with custom loading 
    '''
    
    def __init__(self, 
                 audio_config:AudioConfig,
                 metadata_config:MetadataConfig,
                 dataset_name:str):
        '''
        Initialize AudioDataset
        :param audio_config: AudioConfig object with parameters necessary for loading and transforming audio
        :param metadata_config: MetadataConfig object with parameters necessary for interacting with metadata dataframe
        :param target_col:str, target column in metadata (default = None, sets to task column)
        :param target_labels: List of str, target labels to select from metadata, (default = ['sentence-repetition'])
        '''
        super(AudioDataset, self).__init__()

        self.audio_config = audio_config
        self.metadata_config = metadata_config
        self.metadata = self.metadata_config.metadata.copy()
        self.dataset_name = dataset_name

    def get_metadata(self) -> pd.DataFrame:
        """
        Return metadata
        :return self.metadata: pd.DataFrame
        """
        return self.metadata
    
    def __len__(self) -> int:
        '''
        Get dataset size
        :return: int, len of dataset 
        '''
        return len(self.metadata)
    
    def _get_basic_item(self, idx) -> dict:
        """
        Get basic dataset sample
        :param idx: index as int
        :return: dict, data sample pre-transformation
        """
        uid = self.metadata.index[idx]
        if self.metadata.index.name != self.metadata_config.file_col:
            path = self.metadata[self.metadata_config.file_col].iloc[idx]
        else:
            path = uid

        sample = {
            'uid': uid,
            'path': path,
            'parent_dir': self.audio_config.audio_dir
        }

        return sample
    
    def _get_sa_item(self, idx) -> dict:
        """
        Get Speech Accessibility dataset sample
        :param idx: index as int
        :return: dict, data sample pre-transformation
        """
        uid = self.metadata.index[idx]
        speaker_id = self.metadata[self.metadata_config.speaker_col].iloc[idx]

        sample = {
            'uid': uid,
            'path' : str(Path(speaker_id) / uid),
            'parent_dir': self.audio_config.audio_dir
        }

        return sample
    
    def __getitem__(self, idx:Union[int, torch.Tensor, List[int]]) -> dict:
        """
        Run transformation
        :param idx: index as int
        :return: dict, transformed data sample, Expects that the uid column has been set to index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list): assert len(idx) == 1, 'Should only have one idx even if given as a tensor.'

        if "SpeechAccessibility" in self.dataset_name:
            sample = self._get_sa_item(idx)
        else:
            sample = self._get_basic_item(idx)

        return self.audio_config.transforms(sample)
    
    @staticmethod
    def _get_split_sizes(md_len:int, train:float, val:float) -> Tuple[int, int, int]:
        """
        Get train/val/test sizes based on metadata length

        :param md_len: int, length of metadata
        :param train: float, train set proportion
        :param val: float, validation set proportion
        :return train_sz: int, size of training set
        :return val_sz: int, size of validation set
        :return test_sz: int, size of test set
        """
        train_sz = int(md_len * train) 
        val_sz = int(md_len * val)
        test_sz = md_len - train_sz - val_sz
        return train_sz, val_sz, test_sz 
    
    @staticmethod
    def _new_dataset(sz:int, md:pd.DataFrame, mdc:MetadataConfig, ac:AudioConfig, dn:str) -> AudioDataset: 
        """
        Create a new dataset object
        
        :param sz: int, size of new dataset - if 0, doesn't create a dataset
        :param md: pd.DataFrame, new metadata dataframe
        :param mdc: MetadataConfig, metadata config to update
        :param ac: AudioConfig
        :param dn: str, dataset name
        :return ds: AudioDataset, new audio dataset (none if sz == 0)
        """
        if sz > 0:
            new_config = mdc
            new_config.update_metadata(md)
            ds = AudioDataset(ac, new_config, dn)
        else:
            ds = None
        return ds

    
    def split_by_file(cls, train:float=.7, val:float=.15, seed:int=42):
        """
        Split dataset into train/val/test splits at the individual file level

        :param cls: AudioDataset object
        :param train: float, training proportion
        :param val: float, validation proportion
        :param seed: int, random seed
        """
        assert (train + val) <= 1, 'Train/Val/Test must add up to 1'
        
        all_md = cls.get_metadata().copy()

        train_sz, val_sz, test_sz = cls._get_split_sizes(len(all_md), train, val)
        
        train_md = all_md.sample(train_sz, random_state=seed)
        rem = all_md.drop(train_md.index)
        val_md = rem.sample(val_sz, random_state=seed)
        test_md = rem.drop(val_md.index)

        train_ds = cls._new_dataset(train_sz, train_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        val_ds = cls._new_dataset(val_sz, val_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        test_ds = cls._new_dataset(test_sz, test_md, cls.metadata_config, cls.audio_config, cls.dataset_name)

        return train_ds, val_ds, test_ds

    @staticmethod 
    def _get_speakers(speakers:List[str], beg:int, end:int, md:pd.DataFrame, spk_col:str) -> pd.DataFrame:
        """
        Get subset of speakers in metadata from a list of speakers

        :param speakers: str list, list of all speakers in metadata
        :param beg: int, starting index
        :param end: int, ending index
        :param md: pd.DataFrame, metadata dataframe
        :param spk_col: str, speaker column name
        :return: filtered metadata dataframe (filtered by speaker subset)
        """
        spks = speakers[beg:end]
        return md[md[spk_col].isin(spks)]

    def split_by_speaker(cls, train:float=.7, val:float=.15, seed:int=42):
        """
        Split dataset into train/val/test splits at the speaker level

        :param cls: AudioDataset object
        :param train: float, training proportion
        :param val: float, validation proportion
        :param seed: int, random seed
        """

        assert (train + val) <= 1, 'Train/Val/Test must add up to 1'
        
        all_md = cls.get_metadata().copy()
        speaker_col = cls.metadata_config.speaker_col
        assert speaker_col, 'Metadata must have a speaker ID column to split by speakers.'
        
        speakers = list(set((all_md[speaker_col])))
        train_sz, val_sz, test_sz = cls._get_split_sizes(len(speakers), train, val)
        
        rnd = random.Random(seed)
        rnd.shuffle(speakers)

        train_md = cls._get_speakers(speakers, 0, train_sz, all_md, speaker_col)
        val_md = cls._get_speakers(speakers, train_sz, train_sz+val_sz, all_md, speaker_col)
        test_md = cls._get_speakers(speakers, train_sz+val_sz, len(speakers), all_md, speaker_col)
       
        train_ds = cls._new_dataset(train_sz, train_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        val_ds = cls._new_dataset(val_sz, val_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        test_ds = cls._new_dataset(test_sz, test_md, cls.metadata_config, cls.audio_config, cls.dataset_name)

        return train_ds, val_ds, test_ds
    
    def split_by_diagnosis_speaker(cls, train:float=.7, val:float=.15, seed:int=42):
        """
        Split dataset into train/val/test splits stratified by diagnosis and speaker to ensure equal representation

        :param cls: AudioDataset object
        :param train: float, training proportion
        :param val: float, validation proportion
        :param seed: int, random seed
        """

        assert (train + val) <= 1, 'Train/Val/Test must add up to 1'
        
        train_mds = []
        val_mds = []
        test_mds = []
        all_md = cls.get_metadata().copy()
        speaker_col = cls.metadata_config.speaker_col
        assert speaker_col, 'Metadata must have a speaker ID column to split by speakers.'
        diagnosis_col = cls.metadata_config.diagnosis_col
        assert diagnosis_col, 'Metadata must have a diagnosis column to split by diagnosis.'

        diagnoses = set(list(all_md[diagnosis_col]))
        for d in diagnoses:
            sub_df = all_md[all_md[diagnosis_col] == d]
            sub_speakers = list(set((sub_df[speaker_col])))
            train_sz, val_sz, test_sz = cls._get_split_sizes(len(sub_speakers), train, val)

            rnd = random.Random(seed)
            rnd.shuffle(sub_speakers)

            tr_md = cls._get_speakers(sub_speakers, 0, train_sz, sub_df, speaker_col)
            v_md = cls._get_speakers(sub_speakers, train_sz, train_sz+val_sz, sub_df, speaker_col)
            te_md = cls._get_speakers(sub_speakers, train_sz+val_sz, len(sub_speakers), sub_df, speaker_col)
            train_mds.append(tr_md)
            val_mds.append(v_md)
            test_mds.append(te_md)
        
        train_md = pd.concat(train_mds)
        val_md = pd.concat(val_mds)
        test_md = pd.concat(test_mds)
        train_ds = cls._new_dataset(100*train, train_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        val_ds = cls._new_dataset(100*val, val_md, cls.metadata_config, cls.audio_config, cls.dataset_name)
        test_ds = cls._new_dataset(100-train-val, test_md, cls.metadata_config, cls.audio_config, cls.dataset_name)

        return train_ds, val_ds, test_ds