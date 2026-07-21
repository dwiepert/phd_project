"""

"""
#IMPORTS
##built-in
from pathlib import Path
from collections import defaultdict
import json
from typing import List 
import time

##third-party
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

##local
from sparc import load_model, SpeechWave

class SPARCWrapper():
    """
    checkpoints: en, multi, en+
    device: cpu, cuda:0
    """
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), print_it:bool=True, save_it:bool=True): #checkpoint:str="en", device:str="cpu"):
        super(SPARCWrapper, self).__init__()
        if torch.cuda.is_available():
            d = "cuda:0"
        else:
            d = "cpu"
        self.coder = load_model("en", device=d)
        self.device = device
        self.print_it = print_it
        self.save_it = save_it

    def _sparc_processing(self, wavs:List[torch.tensor]) -> SpeechWave:
        """
        Expects that wavfiles is a list of np.ndarrays
        """
        #assert isinstance(wavfiles, np.ndarray)
        #wavs = [wavfiles]
        wavs = [wav.squeeze().float() for wav in wavs]
        input_lens = np.array([len(wav) for wav in wavs])
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        wavs = SpeechWave(input_values=wavs, input_lens=input_lens)
        wavs = wavs.to(self.device)
        return wavs
    
    def encode_all(self, loader) -> dict:
        """
        """
        for data in tqdm(loader):
            self.encode(data)
            print('pause')

    def encode(self, batch:dict) -> dict:
        """
        """
        output = self._load_ema(batch)
        if output is None:
            if self.print_it:
                print(f'Audio 0 length: {batch['waveform'][0].shape[1] / batch['sample_rate'][0]} s')
            t1 = time.time()
            input = self._sparc_processing(batch['waveform'])
            t2 = time.time()
            if self.print_it:
                print(f'SPARC processing time: {t2-t1}')
            output = self.coder.encode(input)
            if not isinstance(output, list): 
                output = [output]
            t3 = time.time()
            if self.print_it:
                print(f'SPARC encoding time: {t3-t2}')
            if self.save_it:
                self._save_ema(batch, output)
        
        output = self._convert_output(output)

        output_dict = defaultdict(list)
        for o in output:
            for k,v in o.items():
                 output_dict[k].append(v)
        output_dict = dict(output_dict)

        for k in output_dict:
            batch[k] = output_dict[k]
        return batch

    def _load_ema(self, batch:dict) -> dict:
        """
        """
        #check if an _ema.json exists for ALL files in a batch, if not, redoes the whole batch
        paths = batch['path']
        parent_dirs = batch['parent_dir']
        output = []
        for i in range(len(paths)):
            new_path = parent_dirs[i] / Path(paths[i] + "_ema.json")
            if not new_path.exists():
                return None 
            else:
                with open(str(new_path), "r") as f:
                    dict = json.load(f)
                
                output.append(self._unserialize_output(dict))
    
        return output

    def _save_ema(self, batch:dict, output:dict) -> None:
        """
        """
        paths = batch['path']
        parent_dirs = batch['parent_dir']
        for i in range(len(paths)):
            o = output[i]
            new_path = parent_dirs[i] / Path(paths[i] + "_ema.json")
            sub_json = self._serialize_output(o)
         
            with open(str(new_path), "w") as f:
                json.dump(sub_json, f, indent=4)
            
    def _serialize_output(self, o: dict) -> dict:
        """
        """
        new_output = {}
        for k in o:
            if isinstance(o[k], np.ndarray):
                new_output[k] = o[k].tolist()
            elif isinstance(o[k], np.int64):
                new_output[k] = int(o[k])
        return new_output

    def _unserialize_output(self, o:dict) -> dict:
        """
        """
        new_output = {}
        for k in o:
            if isinstance(o[k], list):
                new_output[k] = torch.tensor((o[k]), dtype=torch.float32)
            else:
                new_output[k] = o[k]
        return new_output

    def _convert_output(self, output:List[dict]) -> List[dict]:
        """
        """
        new_output_list = []
        for o in output:
            new_o = {}
            for k in o:
                if isinstance(o[k], np.ndarray):
                    new_o[k] = torch.from_numpy(o[k])
                elif isinstance(o[k], np.int64):
                    new_o[k] = int(o[k])
                else:
                    new_o[k] = o[k]
                new_output_list.append(new_o)
        return new_output_list
            

    