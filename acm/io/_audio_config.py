"""
Audio Configuration class

Author(s): Dani Wiepert 
Last modified: 06/2027
"""

#IMPORTS
#built-in
from pathlib import Path
from typing import Union

#third-party
import torchvision

#local
from acm.constants import _AUDIO_TYPES
from ._resample import ResampleAudio
from ._to_monophonic import ToMonophonic
from ._truncate import Truncate
from ._uid_to_waveform import UidToWaveform
from ._gcs_config import GCSConfig

class AudioConfig():
    def __init__(self, audio_dir:Union[str, Path],
                       audio_ext:str='wav',
                       resample_rate:int=16000,
                       truncate:float=None,
                       gcs_config:GCSConfig=None):
        '''
        Audio config object containing all parameters for loading and transforming audio data
        :param audio_dir: path to audio files (compatible with GCS). Give full path if local and GCS prefix if in cloud
        :param audio_ext: str, audio type (default = 'wav')
        :param resample_rate: int, resample value (default = 16000)
        :param truncate: float, truncate value (default = None)
        :param gcs: GCSConfig object (default = None)
        '''
        super(AudioConfig, self).__init__()
        #GCS compatibility
        self.gcs_config = gcs_config

        #Audio dir
        self.audio_dir = audio_dir
        if self.gcs_config:
            assert isinstance(self.audio_dir, str), 'audio_dir must be given as string prefix for GCS use case.'
        else:
            if not isinstance(self.audio_dir, Path): 
                assert isinstance(self.audio_dir, str), f'Audio directory must be a string or a path but is {type(self.audio_dir)}'
                self.audio_dir = Path(self.audio_dir)

        #Audio extension
        assert isinstance(audio_ext, str), 'Invalid audio extension type.'
        assert audio_ext in _AUDIO_TYPES, f'Invalid audio extension. Must be one of {_AUDIO_TYPES}'
        self.audio_ext = audio_ext

        #Resample rate 
        #assert isinstance(resample_rate, int) or isinstance(int(resample_rate), int), 'Resample rate must be an integer'

        self.resample_rate = int(resample_rate)
       
        #Truncate
        self.truncate = truncate
        if self.truncate: 
            try:
                self.truncate = float(self.truncate)
            except:
                raise TypeError('Truncate must be a float number or string of numbers.')
       
        self._get_transforms()

    def _get_transforms(self):
        self.transforms_list = [UidToWaveform(prefix=self.audio_dir, extension=self.audio_ext, gcs_config=self.gcs_config), 
                                ResampleAudio(resample_rate = self.resample_rate),
                                ToMonophonic()]
        if self.truncate:
            self.transforms_list.append(Truncate(length=self.truncate))
        
        self.transforms = torchvision.transforms.Compose(self.transforms_list)
