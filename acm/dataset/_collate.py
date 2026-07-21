"""
Custom collate function

Author(s): Daniela Wiepert
Last modified: 06/2027
"""
#IMPORTS
##built-in
from typing import List

def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function
    :param batch: input batch, list of samples (dictionaries)
    """
    uid = [item['uid'] for item in batch]
    path = [item['path'] for item in batch]
    parent_dir = [item['parent_dir'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    waveform = [item['waveform'] for item in batch]
    sample = {'uid':uid, 'path':path, 'parent_dir':parent_dir, 'sample_rate':sr, 'waveform':waveform}

    return sample

