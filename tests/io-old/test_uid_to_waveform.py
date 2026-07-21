""" import pytest
from pathlib import Path
from acm.io._uid_to_waveform import UidToWaveform
import torch

### TESTS ###
def test_load_waveform():
    #try with str 
    str_audio_dir = 'tests/data'
    path_audio_dir = Path(str_audio_dir)
    extension = 'wav'
    invalid_extension = 'csv'
    
    #invalid extension
    sample0 = {'uid': 'harvard'}
    with pytest.raises(RuntimeError):
        utw0 = UidToWaveform(prefix=str_audio_dir, extension=invalid_extension)
        utw0(sample0)

    #path doesn't exist
    sample1 = {'uid': 'random'}
    with pytest.raises(RuntimeError):
        utw1 = UidToWaveform(prefix=str_audio_dir, extension=extension)
        utw1(sample1)

    #str path
    #str path
    utw2 = UidToWaveform(prefix=str_audio_dir, extension=extension)
    sample = utw2(sample0)
    assert sample['sample_rate'] == 44100
    assert isinstance(sample['waveform'], torch.Tensor)

    #path
    utw3 = UidToWaveform(prefix=path_audio_dir, extension=extension)
    utw3(sample0)
    assert sample['sample_rate'] == 44100
    assert isinstance(sample['waveform'], torch.Tensor) """