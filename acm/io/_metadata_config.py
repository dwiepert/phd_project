"""
Metadata configuration class

TODO: make compatible with GCS!!!!
Author(s): Dani Wiepert 
Last modified: 06/2027
"""

#IMPORTS
#built-in
from pathlib import Path
from typing import List, Union

#third-party
import pandas as pd

class MetadataConfig():
    def __init__(self, metadata:Union[Path, str, pd.DataFrame],
                       uid_col:str,
                       speaker_col:str=None,
                       diagnosis_col:str = None,
                       file_col:str=None,
                       task_col:str=None,
                       date_col:str=None,
                       session_col:str=None,
                       tasks:List[str] = [],
                       diagnoses:List[str] = []):
        '''
        Metadata config object containing all information for working with metadata dataframe
        
        :param metadata: either path to a metadata or an existing metadata dataframe
        :param uid_col: str, specify which column is the uid col
        :param speaker_col: str, metadata col containing speaker IDs (default = None)
        :param file_col: str, metadata col containing file names (default = None, assumes uid and file col are same unless otherwise specified.)
        :param task_col: str, metadata col containing task names (default = None)
        :param date_col: str, metadata col containing dates (default = None)
        :param session_col: str, metadata col containing session (default = None)
        :param diagnosis_col: str, metadata col containing diagnosis (default = None)
        :param tasks: list[str], list of tasks to filter by (default = [])
        :param diagnoses: list[str], list of diagnoses to filter by (default = [])
        '''
        super(MetadataConfig, self).__init__()
        
        if not isinstance(metadata, pd.DataFrame):
            self._load_metadata(metadata)
        else:
            self.metadata = metadata.copy()

        self.uid_col=uid_col
        self.speaker_col=speaker_col
        self.diagnosis_col=diagnosis_col
        self.file_col=file_col
        self.task_col=task_col
        self.date_col=date_col
        self.session_col=session_col
        self.tasks=tasks
        self.diagnoses=diagnoses

        self._check_columns()

        self._filter()
    
    def _load_metadata(self, metadata:Union[str, Path]) -> pd.DataFrame:
        '''
        Load metadata from file path
        
        :param metadata: path-like, path to metadata csv
        '''
        if not isinstance(metadata, Path):
            assert isinstance(metadata, str), 'Must give string or Path metadata path. '
            metadata=Path(metadata)
        assert metadata.exists(), 'Path to metadata csv does not exist'
        assert metadata.suffix == '.csv', 'Metadata path does not lead to a csv'

        self.metadata = pd.read_csv(metadata) 
    
    def _check_columns(self):
        '''
        Check metadata columns exist in dataframe
        '''
        if self.metadata.index.name != self.uid_col:
            assert self.uid_col in self.metadata, 'uid column must be present in metadata'
            self.metadata = self.metadata.set_index(self.uid_col)

        if self.speaker_col:
            assert self.speaker_col in self.metadata, f'Given speaker id column not in metadata. Please check that the correct speaker_col is specified: {self.speaker_col}'
        if not self.file_col:
            self.file_col = self.uid_col
        assert (self.file_col in self.metadata) or (self.metadata.index.name == self.file_col), f'Given file name column not in metadata. Please check that the correct file_col is specified: {self.file_col}'
        if self.task_col:
            assert self.task_col in self.metadata, f'Given task column not in metadata. Please check that the correct task_col is specified: {self.task_col}'
        if self.date_col:
            assert self.date_col in self.metadata, f'Given date column not in metadata. Please check that the correct date_col is specified: {self.date_col}'
            self.metadata.loc[:,self.date_col] = pd.to_datetime(self.metadata[self.date_col], format='mixed')
        if self.session_col:
            assert self.session_col in self.metadata, f'Given session column not in metadata. Please check that the correct session_col is specified: {self.session_col}'
        if self.diagnosis_col:
            assert self.diagnosis_col in self.metadata, f'Given diagnosis column not in metadata. Please check that the correct diagnosis_col is specified: {self.diagnosis_col}'
    
    def _filter(self):
        '''
        Filter metadata using selected tasks or diagnoses
        '''
        if self.task_col and self.tasks:
            self.metadata = self.metadata[self.metadata[self.task_col].isin(self.tasks)]
            assert not self.metadata.empty, 'Selected tasks not present in metadata'
        if self.diagnosis_col and self.diagnoses:
            self.metadata = self.metadata[self.metadata[self.diagnosis_col].isin(self.diagnoses)]
            assert not self.metadata.empty, 'Selected diagnoses not present in metadata'

    def update_metadata(self, new_metadata:Union[str, Path, pd.DataFrame]):
        '''
        Update metadata to new_metadata
        :param new_metadata: new metadata data path or dataframe
        '''
        if not isinstance(new_metadata, pd.DataFrame):
            self._load_metadata(new_metadata)
        else:
            self.metadata = new_metadata.copy()
        self._check_columns()
        self._filter()
