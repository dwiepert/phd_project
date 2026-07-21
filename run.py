"""
Run file

Author(s): Dani Wiepert
Last modified: 06/2027
"""

#IMPORTS
##built-in
import argparse
from pathlib import Path

##third-party
import torch
from torch.utils.data import DataLoader

##local
from acm.dataset import *
from acm.io import *
from acm.constants import *
from acm.modules import SPARCWrapper

def zip_gcs_config(args:argparse.Namespace):
    """
    TODO
    """
    gcs_args = {'bucket_name': args.bucket_name, 'project_name': args.project_name}
    return gcs_args

def zip_audio_config(args:argparse.Namespace):
    """
    TODO
    """
    audio_args = {'audio_dir':args.audio_dir, 
                  'audio_ext':args.audio_ext, 
                  'resample_rate': args.resample_rate,
                  'truncate':args.truncate}
    return audio_args

def zip_metadata_config(args:argparse.Namespace):
    """
    TODO
    """
    metadata_args = {'metadata': args.metadata_path}

    assert args.dataset_name in _DATASET_PARAMS, f'{args.dataset_name} is not an implemented dataset.'
    params = _DATASET_PARAMS[args.dataset_name]['params']
    metadata_args = metadata_args | params

    assert set(args.tasks).issubset(_DATASET_PARAMS[args.dataset_name]['tasks'])
    metadata_args['tasks'] = args.tasks

    assert set(args.diagnoses).issubset(_DATASET_PARAMS[args.dataset_name]['diagnoses'])
    metadata_args['diagnoses'] = args.diagnoses
    return metadata_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Parser")
    #PATHS
    parser.add_argument("--audio_dir", type=Path, help="Path to audio files.")
    parser.add_argument("--metadata_path", type=Path, help="Path to metadata csv.")
    parser.add_argument("--ema_only", action="store_true")
    #AUDIO ARGS
    parser.add_argument("--audio_ext", type=str, default='wav', help="Audio type. Default = wav.")
    parser.add_argument("--resample_rate", type=int, default=16000, help="Resampling rate. Default = 16000.")
    parser.add_argument("--truncate", type=float, help="Specify amount of time (in s) to truncate from an audio file.")
    #DATA ARGS
    parser.add_argument("--dataset_name", type=str, default="SpeechAccessibility_2026-04-08", choices=_DATASETS, help="Specify dataset name. Default = SpeechAccessibility_2026-04-08")
    parser.add_argument("--tasks", nargs="+", default=_DATASET_PARAMS["SpeechAccessibility_2026-04-08"]['tasks'], help=f"Specify target tasks for dataset.")
    parser.add_argument("--diagnoses", nargs="+", default=_DATASET_PARAMS["SpeechAccessibility_2026-04-08"]['diagnoses'], help=f"Specify target tasks for dataset.")
    parser.add_argument("--split", choices=["file", "speaker", "speaker+diagnosis"], help="Specify whether to split dataset by randomly sampling audio files.")
    parser.add_argument("--train_proportion", type=float, default=0.7, help="Specify training proportion. Default = 0.7.")
    parser.add_argument("--val_proportion", type=float, default=0.15, help="Specify validation proportion. Default = 0.7.")
    #GCS ARGS
    parser.add_argument("--bucket_name", type=str, help="GCS bucket name")
    parser.add_argument("--project_name", type=str, help="GCS project name")
    #TRAINING
    parser.add_argument("--batch_sz", type=int, default=1, help="Batch size for data loading. Default = 1.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader. Default = 0.")
    
    #OTHER
    parser.add_argument("--seed", type=int, default=100, help="Random seed. Default = 100.")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #GCS Config
    if args.bucket_name:
        gcs_args = zip_gcs_config(args)
        gcs_config = GCSConfig(**gcs_args)
    else:
        gcs_config=None

    #Audio Config
    ac_args = zip_audio_config(args)
    ac = AudioConfig(gcs_config=gcs_config, **ac_args)

    #Metadata Config
    mc_args = zip_metadata_config(args)
    mc = MetadataConfig(**mc_args)

    #initial dataset
    audio_dataset = AudioDataset(audio_config=ac, metadata_config=mc, dataset_name=args.dataset_name)

    if args.ema_only: 
        loader = DataLoader(dataset=audio_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)
        w = SPARCWrapper(args.device, args.print, True)
        print('EMA encoding ...')
        w.encode_all(loader)

    else:
        if args.split:
            if args.split == "file":
                train, val, test = AudioDataset.split_by_file(audio_dataset, args.train_proportion, args.val_proportion, args.seed)
            elif args.split == "speaker":
                train, val, test = AudioDataset.split_by_speaker(audio_dataset, args.train_proportion, args.val_proportion, args.seed)
            elif args.split == "speaker+diagnosis":
                train, val, test = AudioDataset.split_by_diagnosis_speaker(audio_dataset, args.train_proportion, args.val_proportion, args.seed)
            else:
                raise NotImplementedError(f"{args.split} not a valid splitting method")

            if args.debug: 
                train_shuffle = False
                args.save_it = True
            else:
                train_shuffle = True
                
            if train:
                train_loader = DataLoader(dataset=train,batch_size=args.batch_sz,shuffle=train_shuffle,collate_fn=collate_fn, num_workers=args.num_workers)
            if val:
                val_loader = DataLoader(dataset=val,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)
            if test:
                test_loader = DataLoader(dataset=test,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)

    #     #TODO: test loader works for different batch sizes
    #     batch = next(iter(train_loader))
        
    #     if args.debug:
    #         for i in range(10):
    #             w = SPARCWrapper()
    #             w.encode(batch)
    #     #coder = load_model("en", device="cpu")
    #     #coder.encode(batch['waveform'])
    #         print('pause')
    #     else:
    #         w = SPARCWrapper()
    #         w.encode(batch)
    #     #TODO: modules
    # else:
    #     loader = DataLoader(dataset=audio_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)
    #     batch = next(iter(loader))
    #     print('pause')
    #     #TODO: modules
