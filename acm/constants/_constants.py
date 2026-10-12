"""
Various constants for ensuring consistency
Author(s): Dani Wiepert
Last modified: 06/2026
"""

_AUDIO_TYPES = {'wav', 'mp3', 'flac', 'ogg', 'aac', 'opus', 'aiff', 'au'}
_WAVLM_STRIDES = (5, 2, 2, 2, 2, 2, 2)
_SPARC_SENSORS = ['td_x','td_y','tb_x','tb_y','tt_x','tt_y','li_x','li_y','ul_x','ul_y','ll_x','ll_y']

#['ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y', 'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y']

['TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY']
def _wavlm_frame_hop_seconds():
        total_stride = 1
        for s in _WAVLM_STRIDES:
            total_stride *= s
        return total_stride / 16000.0  # 320/16000 = 0.02s = 20ms

_WAVLM_HOP = _wavlm_frame_hop_seconds()

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
                    'mocha-timit': {'params': {'uid_col': 'uid',
                                               'speaker_col': 'speaker_id', 
                                               'file_col': None,
                                               'session_col':None,
                                               'date_col': None,
                                               'task_col': None,
                                               'tasks': None},
                                               'sample_rate':500,
                                               'time_col': 'time',
                                               'ema_pattern': '*.ema',
                                               'sensors': ['ui_x', 'li_x', 'ul_x', 'll_x', 'tt_x', 'ui_y', 'li_y', 'ul_y', 'll_y', 'tt_y', 'tb_x', 'td_x', 'v_x', 'bn_x', 'tb_y', 'td_y', 'v_y','bn_y'],
                                               'exclude_cols': ['time', 'present'],
                                               'exclude_sensors': ['ui_x', 'ui_y', 'v_x', 'v_y', 'bn_x', 'bn_y'],
                                               'tasks':None,
                                               'diagnoses':None},

                    'sparc': { 'sample_rate':50,
                                'time_col': None,
                                'ema_pattern': '*_ema.json',
                                'sensors': _SPARC_SENSORS,
                                'exclude_cols': [],
                                'exclude_sensors': [],
                                'tasks':None,
                                'diagnoses':None},

                    'als': {'params': {'uid_col': 'uid',
                                               'speaker_col': 'speaker_id', 
                                               'file_col': None,
                                               'session_col':None,
                                               'date_col': None,
                                               'task_col': None,
                                               'tasks': None},
                            'tasks':None,
                            'diagnoses':None},

                    'torgo': {'params': {'uid_col': 'uid',
                                        'speaker_col': 'speaker', 
                                        'file_col': None,
                                        'session_col':'session',
                                        'date_col': None,
                                        'task_col': None,
                                        'tasks': None},
                                'tasks':None,
                                'diagnoses':None},
                                           
                    'mngu0': {'params': {'uid_col': 'uid',
                                            'speaker_col': 'speaker', 
                                            'file_col': None,
                                            'session_col':'session',
                                            'date_col': None,
                                            'task_col': None,
                                            'tasks': None},
                                'tasks':None,
                                'diagnoses':None},

                    'MayoDataset':  {'params': {'uid_col': 'uid',
                                                'speaker_col':'speakerID',
                                                'file_col': 'file_name',
                                                'task_col': 'task',
                                                'date_col': 'date',
                                                'session_col': 'session'},
                                                'tasks': ['sentence-repetition'],
                                                'diagnoses':None}}

_DATASETS = list(_DATASET_PARAMS)
