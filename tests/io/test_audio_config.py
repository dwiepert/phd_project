"""
Test Audio Configuration

Author(s): Dani Wiepert
Last modified: 06/2026
"""
#IMPORTS
##built-in 
from pathlib import Path

##third-party 
import pytest

##local
from acm.io import AudioConfig, GCSConfig

def test_audio_config(data_dir):
    #NO GCS
    #Audio dir not proper type
    with pytest.raises(AssertionError):
        AudioConfig(audio_dir=16000)

    #Audio dir saved as path
    A1 = AudioConfig(audio_dir=str(data_dir))
    assert isinstance(A1.audio_dir, Path)
    A2 = AudioConfig(audio_dir=data_dir)
    assert isinstance(A2.audio_dir, Path)

    #Incompatible audio extension
    with pytest.raises(AssertionError):
        AudioConfig(audio_dir=data_dir, audio_ext='random')
    
    with pytest.raises(AssertionError):
        AudioConfig(audio_dir=data_dir, audio_ext=123)

    #invalid resample rate
    with pytest.raises(ValueError):
        AudioConfig(audio_dir=data_dir, resample_rate='audio')
    Arr = AudioConfig(audio_dir=data_dir, resample_rate='16000') 

    #invalid truncate
    with pytest.raises(TypeError):
        AudioConfig(audio_dir=data_dir,truncate='audio')
    
    A3 = AudioConfig(audio_dir=data_dir,truncate=1)
    A4 = AudioConfig(audio_dir=data_dir,truncate=.5)
    A5 = AudioConfig(audio_dir=data_dir,truncate='123')
    #With GCS 
    #TODO

