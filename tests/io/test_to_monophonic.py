import pytest
from acm.io._uid_to_waveform import UidToWaveform
from acm.io._to_monophonic import ToMonophonic

### TESTS ###
def test_monophonic_waveform():
    str_audio_dir = 'tests/data'
    
    sample = {'uid': 'harvard'}
    utw = UidToWaveform(str_audio_dir)
    wav_sample = utw(sample)
    wav = wav_sample['waveform']
    assert wav.shape[0] == 2
    m = ToMonophonic()
    m_sample = m(wav_sample)
    assert m_sample['waveform'].shape[0] == 1
