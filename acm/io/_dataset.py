"""
Custom Dataset implementation 

Author(s): Dani Wiepert 
Last modified: 01/2026
"""

#IMPORTS 
##built-in 
from pathlib import Path
from typing import List, Union, Tuple

##third-party
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
import torchvision

##local 
from ._uid_to_waveform import UidToWaveform
from ._to_monophonic import ToMonophonic
from ._resample import ResampleAudio
from ._truncate import Truncate

class CustomDataset(Dataset):
    """
    Custom audio dataset implementation 
    """

    def __init__(self, audio_dir:Union[Path, str], metadata_path:Union[Path, str], audio_ext:str='wav',
                 uid_col:str='uid', audio_col:str='file_name', speaker_col:str='speakerID', task_col:str='task', tasks:List[str]=['sentence-repetition'],
                 resample_rate:int=16000, truncate:int=None, debug:bool=False):
        """
        :param audio_dir: path-like, path to audio directory
        :param metadata_path: path-like, path to metadata csv
        :param audio_ext: str, audio extension (default = 'wav')
        :param uid_col: str, metadata col containing unique IDs (default = 'uid')
        :param audio_col: str, metadata col containing audio file names (default = 'file_name')
        :param speaker_col: str, metadata col containing speaker IDs (default = 'speaker_col')
        :param task_col: str, metadata col containing task names (default = 'tasks')
        :param tasks: List of str, tasks to select, (default = ['sentence-repetition'])
        :param resample_rate: int, resampling rate (default = 16000)
        :param truncate: int, amount to truncate audio in seconds (default = None)
        :param audio_ext: str, audio extension (default = 'wav')
        """
        super(CustomDataset, self).__init__()

        # AUDIO DIRECTORY
        self.audio_dir = audio_dir
        if not isinstance(self.audio_dir, Path): self.audio_dir = Path(self.audio_dir)
        self.audio_ext = audio_ext
        assert self.audio_dir.exists(), f'{str(self.audio_dir)} does not exist. Please check directory and try again.'
        assert [f for f in self.audio_dir.rglob(pattern=f'*.{self.audio_ext}')] != [], f'Not {self.audio_ext} files exist in given audio directory.'

        # METADATA PATH
        self.metadata_path = metadata_path
        if not isinstance(self.metadata_path, Path): self.metadata_path = Path(self.metadata_path)
        assert self.metadata_path.exists(), f'{str(self.metadata_path)} does not exist. Please check metadata file name and try again.'
        assert self.metadata_path.suffix == '.csv', f'Metadata expected to be a csv file but is {self.metadata_path.suffix}. Please convert and retry.'
        self.metadata = pd.read_csv(self.metadata_path)

        # CHECK COLUMNS ARE AS EXPECTED IN METADATA
        self.uid_col = uid_col
        self.audio_col = audio_col
        self.speaker_col = speaker_col 
        self.task_col = task_col 

        self._check_metadata()

        #Filter by tasks
        self.tasks = tasks
        self.metadata = self.metadata[self.metadata[self.task_col].isin(self.tasks)]

        # AUDIO TRANSFORMS
        self.resample_rate = resample_rate
        self.truncate = truncate
        self.debug = debug
        if self.debug:
            self.metadata = self.metadata.sample(n=100)
        
        self.transforms_list = [UidToWaveform(prefix=self.audio_dir, extension=self.audio_ext), 
                                ResampleAudio(resample_rate = self.resample_rate),
                                ToMonophonic()]
        if self.truncate:
            self.transforms_list.append(Truncate(length=self.truncate))
        
        self.transforms = torchvision.transforms.Compose(self.transforms_list)

    def _check_metadata(self):
        """
        Check that the metadata is valid
        """
        if self.metadata.index.name != self.uid_col:
            assert self.uid_col in self.metadata, 'uid column must be present in metadata'
            self.metadata = self.metadata.set_index(self.uid_col)

        assert self.audio_col in self.metadata, f'Given audio id column not in metadata. Please check that the correct audio_col is specified: {self.audio_col}'
        assert self.speaker_col in self.metadata, f'Given speaker id column not in metadata. Please check that the correct speaker_col is specified: {self.speaker_col}'
        assert self.task_col in self.metadata, f'Given task column not in metadata. Please check that the correct task_col is specified: {self.task_col}'

    def update_metadata(self, new_metadata:pd.DataFrame):
        """
        Update metadata
        """
        assert isinstance(new_metadata, pd.DataFrame), 'Must give a pandas dataframe as new metadata'
        self.metadata = new_metadata
        self._check_metadata()
        self.metadata = self.metadata[self.metadata[self.task_col].isin(self.tasks)]

    def get_metadata(self) -> pd.DataFrame:
        """
        Return metadata
        :return self.metadata: pd.DataFrame
        """
        return self.metadata

    def __len__(self) -> int:
        """
        Get dataset size
        :return: int, len of dataset 
        """
        return len(self.metadata)
    
    def __getitem__(self, idx:Union[int, torch.Tensor, List[int]]) -> dict:
        """
        Run transformation
        :param idx: index as int
        :return: dict, transformed data sample, Expects that the uid column has been set to index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list): assert len(idx) == 1, 'Should only have one idx even if given as a tensor.'
    
        uid = self.metadata.index[idx]

        sample = {
            'uid' : uid
        }

        return self.transforms(sample)

    @classmethod
    def split_data(cls, train_ratio:float=0.7, val_ratio:float=0.15, existing_split:Union[Path, str]=None, seed:int=100) -> tuple:
        """
        Create data splits

        :param train_ratio: float, train ratio for split (default = 0.7)
        :param val_ratio: float, val_ratio for split (default = 0.15)
        :param seed: int, random seed (default=100)
        """
        torch.manual_seed(seed)

        test_ratio = 1 - train_ratio - val_ratio
        assert (train_ratio + val_ratio + test_ratio) == 1, 'train/val/test ratios must sum to 1.'

        metadata = cls.metadata.copy()


        #TODO: split
        ## LOAD EXISTING SPLIT

        ## CREATE NEW SPLIT

        train_md = None 
        val_md = None 
        test_md = None 

        ## MAKE CLS COPIES AND UPDATE
        train_cls = cls.copy()
        train_cls.update_metadata(train_md)

        val_cls = cls.copy()
        val_cls.update_metadata(val_md)

        test_cls = cls.copy()
        test_cls.update_metadata(test_md)

        return train_cls, val_cls, test_cls
    
