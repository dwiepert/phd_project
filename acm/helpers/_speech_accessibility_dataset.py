
from pathlib import Path
import os 
import shutil
import json
from collections import defaultdict
import pandas as pd

def unzip(directory):
    """
    """
    if not isinstance(directory, Path): directory = Path(directory)
    files = [f for f in directory.rglob("*.tar")]
    unexpandable = []
    for i in range(len(files)):
        f =files[i]
        if i % 50 == 0: print(f'{i} out of {len(files)}')
        try:
            shutil.unpack_archive(str(f), str(f.parent))
            os.remove(f)
        except:
            unexpandable.append(f)

    return unexpandable

def move_up_1(directory):
    if not isinstance(directory, Path): directory = Path(directory)
    files = [f for f in directory.rglob("*") if not f.is_dir() and f.suffix in ['.wav', '.json']]
    for f in files:
        parts = list(f.parts)
        if parts[-3] != 'Dev' and parts[-3]!= 'Train':
            del parts[-3]
            new_path = Path(*parts[:-1])
            new_path.mkdir(parents=True, exist_ok=True)
            shutil.move(f, new_path)
        else:
            continue

def create_metadata_SA(directory):
    if not isinstance(directory, Path): directory = Path(directory)
    files = [f for f in directory.rglob("*.json")]
    md = []
    unreadable = [] 
    for f in files:
        try:
            with open(str(f)) as j:
                d = json.load(j, strict=False)
                sub_dict = {'speaker_id': f.parent.name,
                            'etiology': d['Etiology']}
                
                finfo = d['Files']
                for fi in finfo:
                    file_dict = {'uid': fi['Filename'].split(".")[0], 'date':fi['Created'], 'prompt_text':fi['Prompt']['Prompt Text'], 
                                'transcript':fi['Prompt']['Transcript'], 'category':fi['Prompt']['Category Description'],
                                'subcategory': fi['Prompt']['Sub Category Description'],
                                'ratings': fi['Ratings']}
                    combined_dict = file_dict | sub_dict
                    md.append(combined_dict)
        except:
            unreadable.append(f)
    all_md = defaultdict(list)
    for d in md:
        for key, value in d.items():
            all_md[key].append(value)
    metadata_dict = dict(all_md)
    metadata_df = pd.DataFrame(metadata_dict)
    md_path = directory / 'metadata.csv'
    metadata_df.to_csv(md_path)
    return unreadable


directory = "/Users/dwiepert/Documents/SCHOOL/Grad_School/SARR_dataset/Downsampled/Train"

#unexpandable = unzip(directory)
#move_up_1(directory)
unreadable = create_metadata_SA(directory)
print('pause')
