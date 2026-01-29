import pytest
from acm.io._uid_to_waveform import UidToWaveform
from acm.io._resample import ResampleAudio

### TESTS ###
def test_resample_waveform():
    str_audio_dir = 'tests/data'
    
    sample = {'uid': 'harvard'}
    utw = UidToWaveform(str_audio_dir)
    wav_sample = utw(sample)
    len_pre = wav_sample['waveform'].shape[1]
    r = ResampleAudio(resample_rate = 16000)
    r_sample = r(wav_sample)
    assert r_sample['sample_rate'] == 16000
    len_post = r_sample['waveform'].shape[1]
    assert len_pre > len_post
