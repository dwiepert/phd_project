import pytest
from acm.io._uid_to_waveform import UidToWaveform
from acm.io._truncate import Truncate
import torch 

### TESTS ###
def test_resample_truncate(data_dir):
    str_audio_dir = str(data_dir)
    
    sample = {'uid': 'harvard', 'path': 'harvard'}
    utw = UidToWaveform(str_audio_dir)
    wav_sample = utw(sample)
    len_pre = wav_sample['waveform'].shape[1]
    
    #test length
    t0 = Truncate(length=1)
    t0_sample = t0(wav_sample)
    assert t0_sample['waveform'].shape[1] == t0_sample['sample_rate']


    #test offset
    t1 = Truncate(length=1, offset=.2)
    start = int(.2*wav_sample['sample_rate'])
    t1_sample = t1(wav_sample)
    assert (t1_sample['waveform'].shape[1] == t1_sample['sample_rate']) and (torch.equal(wav_sample['waveform'][:,start],t1_sample['waveform'][:,0]))
    

    #test padding mean
    max_len = int(len_pre / wav_sample['sample_rate'])
    frames = int((max_len + 2) * wav_sample['sample_rate'])
    n_remaining = int(frames - len_pre)
    t2 = Truncate(length=max_len+2,pad=True,pad_method='mean')
    t2_sample = t2(wav_sample)
    assert t2_sample['waveform'].shape[1] == frames 
    assert t2_sample['waveform'][0,-n_remaining] != 0

    #test padding 0
    t3 = Truncate(length=max_len+2,pad=True,pad_method='zero')
    t3_sample = t3(wav_sample)
    assert t3_sample['waveform'].shape[1] == frames 
    assert t3_sample['waveform'][0,-n_remaining] == 0
