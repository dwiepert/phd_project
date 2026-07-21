import pytest
from pathlib import Path
from acm.io._uid_to_waveform import UidToWaveform
from acm.io._truncate import Truncate
import torch 

def test_init(data_dir):
    """
    """
    uw0 = UidToWaveform(prefix=str(data_dir))
    uw1 = UidToWaveform(prefix=Path(data_dir))

    with pytest.raises(TypeError):
        suw = UidToWaveform(prefix=1)
    
    with pytest.raises(AssertionError):
        uw = UidToWaveform(prefix=data_dir, extension='random')

    #TODO: test gcs_config

def test_call(data_dir):
    """
    """
    uw0 = UidToWaveform(prefix=str(data_dir))
    #TODO test gcs load
    #test uid in cache
    sample = {'uid': 'harvard', 'path': 'harvard'}
    wav_sample = uw0(sample)
    assert 'waveform' in wav_sample
    assert 'sample_rate' in wav_sample
    assert 'path' in wav_sample 
    assert sample['uid'] in uw0.cache
    cache = uw0.cache[sample['uid']]
    assert torch.equal(cache['waveform'],wav_sample['waveform'])
    assert cache['sample_rate'] == wav_sample['sample_rate']
    assert cache['path'] == wav_sample['path']
    sample2 = {'uid': 'random', 'path': 'random'}
    
    with pytest.raises(RuntimeError):
        w = uw0(sample2)
    