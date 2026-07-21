"""
Load waveform from a uid

Author(s): Dani Wiepert
Last modified: 06/2026
"""
#IMPORTS
##built-in
import io
from typing import Union, Tuple
from pathlib import Path

##third party
import torch
import torchaudio

##local
from ._gcs_config import GCSConfig
from acm.constants import _AUDIO_TYPES

class UidToWaveform(object):
    '''
    Take a UID, find & load the data, add waveform and sample rate to sample
    :param prefix:str, path prefix for searching
    :param extension:str, audio file extension (default = wav)
    :param gcs_config: GCSConfig object (default = None)
    '''
    
    def __init__(self, prefix:Union[str,Path], extension:str='wav', gcs_config:GCSConfig=None):
        
        self.prefix = prefix #input_dir prefix
        if not isinstance(self.prefix, Path): self.prefix = Path(self.prefix)
        self.cache = {}
        self.extension = extension
        assert self.extension in _AUDIO_TYPES, f'{self.extension} is not a valid audio type. Choose one of {_AUDIO_TYPES}.'
        self.gcs_config = gcs_config

        if self.gcs_config is None:
            assert self.prefix.exists(), 'Path must exist if loading waveform.'
        else:
            assert self.gcs_config.exists(self.prefix), 'Path must exist if loading waveform.'

    def _load_waveform(self, sub_path:str) -> Tuple[torch.tensor, int]:
        '''
        Load waveform
        '''
        waveform_path = self.prefix / f'{sub_path}.{self.extension}'

        if self.gcs_config:
            blob = self.gcs_config.get_blob(str(waveform_path))
            wave_input = io.BytesIO(blob.download_as_bytes())
        else:
            wave_input = waveform_path

        waveform, sr = torchaudio.load(wave_input)
        return waveform, sr
    

    def __call__(self, sample:dict) -> dict:
        """
        Load waveform
        :param sample: dict, input sample
        :return wavsample: dict, sample after loading
        """
        wavsample = sample.copy()
        sub_path = wavsample['path']
        uid = wavsample['uid']
        cache = {}
        if uid not in self.cache:
            wav, sr = self._load_waveform(sub_path)
            cache['waveform'] = wav 
            cache['sample_rate'] = sr
            cache['path'] = sub_path
            self.cache[uid] = cache
            
        cache = self.cache[uid]
        
        wavsample['waveform'] = cache['waveform']
        wavsample['sample_rate'] = cache['sample_rate']
        wavsample['path'] = cache['path']
         
        return wavsample