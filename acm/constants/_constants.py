"""
Various constants for ensuring consistency
Author(s): Dani Wiepert
Last modified: 06/2026
"""

_AUDIO_TYPES = {'wav', 'mp3', 'flac', 'ogg', 'aac', 'opus', 'aiff', 'au'}

_DATASET_PARAMS = {'SpeechAccessibility_2026-04-08': {'params': {'uid_col': 'uid',
                                                      'speaker_col':'speaker_id',
                                                      'file_col': 'uid',
                                                      'task_col': 'category',
                                                      'date_col': 'date',
                                                      'session_col': None,
                                                      'diagnosis_col': 'etiology'},
                                                      'tasks': ['Digital Assistant Commands', 
                                                                'Spontaneous Speech Prompts', 
                                                                'Non-spontaneous Speech Prompts', 
                                                                'Novel Sentences'],
                                                      'diagnoses': ['ALS', 
                                                                    'Cerebral Palsy', 
                                                                    'Down Syndrome', 
                                                                    'Parkinson\'s Disease', 
                                                                    'Stroke']},
                    'MayoDataset':  {'params': {'uid_col': 'uid',
                                                'speaker_col':'speakerID',
                                                'file_col': 'file_name',
                                                'task_col': 'task',
                                                'date_col': 'date',
                                                'session_col': 'session'},
                                                'tasks': ['sentence-repetition']}}

_DATASETS = list(_DATASET_PARAMS)