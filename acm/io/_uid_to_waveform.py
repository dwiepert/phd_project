"""
Load waveform from a uid

Author(s): Dani Wiepert
Last modified: 01/2026
"""
#IMPORTS
##built-in
from typing import Union, Tuple
from pathlib import Path

##third party
import torch
import torchaudio

class UidToWaveform(object):
    '''
    Take a UID, find & load the data, add waveform and sample rate to sample
    :param prefix:str, path prefix for searching
    :param extension:str, audio file extension (default = None)
    :param structured: bool, indicate whether audio files are in structured format (prefix/uid/waveform.wav) or not (default=False)
    '''
    
    def __init__(self, prefix:Union[str,Path], extension:str='wav'):
    
        self.prefix = prefix #input_dir prefix
        if not isinstance(self.prefix, Path): self.prefix = Path(self.prefix)
        self.cache = {}
        self.extension = extension
    
    def _load_waveform(self, uid:str) -> Tuple[torch.tensor, int]:
        '''
        Load waveform
        '''
        waveform_path = self.prefix 
        waveform_path = waveform_path / f'{uid}.{self.extension}'
        waveform, sr = torchaudio.load(waveform_path)
        return waveform, sr

    def __call__(self, sample:dict) -> dict:
        """
        Load waveform
        :param sample: dict, input sample
        :return wavsample: dict, sample after loading
        """
        wavsample = sample.copy()
        uid = wavsample['uid']
        cache = {}
        if uid not in self.cache:
            wav, sr = self._load_waveform(uid)
            cache['waveform'] = wav 
            cache['sample_rate'] = sr
            self.cache[uid] = cache
            
        cache = self.cache[uid]
        
        wavsample['waveform'] = cache['waveform']
        wavsample['sample_rate'] = cache['sample_rate']
         
        return wavsample
    
