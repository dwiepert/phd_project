"""
Functions for processing EMA data
"""
#IMPORTS
##built-in
from collections import defaultdict
from typing import Union, List
from pathlib import Path

##third-party
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

##local
from acm.modules import SPARC
from acm.constants import *

class EMA():
    """
    """
    def __init__(self, ema_dir:Union[str,Path], wrapper:Union[SPARC], dataset:str,  downsample_frames:List[int]=None, normalize_method:str='zscore', downsample_method:str='wavlm', downsample_factor:int=10, per_speaker:bool=True, per_session:bool=False):
        self.ema_dir = Path(ema_dir)
        self.dataset = dataset
        self.wrapper = wrapper 
        if self.dataset != 'sparc':
            assert downsample_frames is not None, 'must give sparc frames if EMA is not sparc already.'
            self.downsample_frames = downsample_frames
            self.normalize_method = normalize_method
            self.downsample_method = downsample_method
            self.per_speaker = per_speaker
            self.per_session = per_session
            self.downsample_factor = downsample_factor
        else:
            self.normalize_method = 'none'
            self.downsample_method = 'none'
            self.per_speaker = False
            self.per_session = False
            self.downsample_factor = None

        self.sensors = _DATASET_PARAMS[self.dataset]['sensors']
        self.exclude_cols = _DATASET_PARAMS[self.dataset]['exclude_cols']
        self.exclude_sensors = _DATASET_PARAMS[self.dataset]['exclude_sensors']
        self.exclude = self.exclude_cols + self.exclude_sensors 
        self.ema_pattern = _DATASET_PARAMS[self.dataset]['ema_pattern']
        
        self._clean_ema_files()
        if self.dataset == 'sparc':
            self.downsample_frames = {p: len(df) for p, df in self.cleaned_data.items()}
        else:
            self._chan_cols = [c for c in next(iter(self.cleaned_data.values())).columns if c not in self.exclude]
            self.per_file_stats = {p: self._compute_stats(df) for p, df in self.cleaned_data.items()}
            self._merge_stats()

        self._normalize()
        self._downsample()

    def _clean_ema_files(self):
        """
        """
        self.ema_files = [e for e in self.ema_dir.rglob(self.ema_pattern)]
        self.cleaned_data = {}
        self.errors = {}
        self.speakers = {}
        self.session = {}
        #self.num_frames = {}
        for e in self.ema_files:
            name = self.wrapper.get_name(e)
            self.speakers[name] = self.wrapper.get_speaker(e)
            self.session[name] = self.wrapper.get_session(e)

            df = self.wrapper.load_ema(e)
            if self.dataset == 'sparc':
                df = self.wrapper.to_df(df)
            
            for c in self.sensors:
                if c not in self.exclude_sensors:
                    if c not in df:
                        self.errors[name] = df
                        continue
            if name in self.errors: continue 

            data = self.wrapper.clean_ema(df)
            self.cleaned_data[name] = data
            #self.num_frames[name] = data.to_numpy().shape[0]

    def _compute_stats(self, df:pd.DataFrame):
        """
        EMA stats
        :param df: pd.DataFrame, ema values
        :param chan_cols: list of coil columns
        :return: dict of stats
        """
        all = {}
        for c in self._chan_cols:
            col = df[c]
            me = float(col.mean())
            std = float(col.std() or 1.0)
            mi = float(col.min())
            ma = float(col.max())
            all[c] = {"mean":me, "std": std,
                      "min": mi, "max": ma}

        return all

    def _merge_stats(self):
        """
        Combine per-file stats into overall mean/std/min/max (for --stats-file).
        :param stats_lst: list of stats
        :param chan_cols: list of coil columns
        :return merged: dictionary of merged stats
        """
        self.merged_stats = {}
        grouped = defaultdict(list)
        if self.per_speaker and self.per_session:
            for name, spk in self.speakers.items():
                if name in self.session:
                    ssn = self.session[name]
                    spk_ssn = f'{spk}_{ssn}'
                    if name in self.per_file_stats:
                        grouped[spk_ssn].append(name)

        elif self.per_speaker:
            for name, spk in self.speakers.items():
                if name in self.per_file_stats:
                    grouped[spk].append(name)
        else:
            for name, _ in self.cleaned_data.items():
                grouped['files'].append(name)

        for group, files in grouped.items():
            merged = {}
            for c in self._chan_cols:
                ns = np.array([1.0 for f in files])
                means = np.array([self.per_file_stats[f][c]["mean"] for f in files])
                stds = np.array([self.per_file_stats[f][c]["std"] for f in files])
                mins = np.array([self.per_file_stats[f][c]["min"] for f in files])
                maxs = np.array([self.per_file_stats[f][c]["max"] for f in files])
        
                total_n = ns.sum()
                pooled_mean = float(means.sum() /len(files))
                pooled_var = float((stds ** 2 + (means - pooled_mean) ** 2).sum() / len(files))
        
                merged[c] = {
                    "mean": pooled_mean,
                    "std": float(np.sqrt(pooled_var)) or 1.0,
                    "min": float(mins.min()),
                    "max": float(maxs.max()),
                    "n_files": len(files)
                }

            for f in files:
                self.merged_stats[f] = merged


        """
        speaker_stats = {}
    for spk, files in grouped.items():
        channels = stats[files[0]].keys()
        merged = {}
        for ch in channels:
            ns = np.array([float(counts[f]) if counts else 1.0 for f in files])
            means = np.array([stats[f][ch]["mean"] for f in files])
            stds = np.array([stats[f][ch]["std"] for f in files])
            mins = np.array([stats[f][ch]["min"] for f in files])
            maxs = np.array([stats[f][ch]["max"] for f in files])
 
            total_n = ns.sum()
            pooled_mean = float((ns * means).sum() / total_n)
            pooled_var = float((ns * (stds ** 2 + (means - pooled_mean) ** 2)).sum() / total_n)
 
            merged[ch] = {
                "mean": pooled_mean,
                "std": float(np.sqrt(pooled_var)) or 1.0,
                "min": float(mins.min()),
                "max": float(maxs.max()),
                "n_files": len(files),
                "n_frames": float(total_n) if counts else None,
            }
        speaker_stats[spk] = merged
        """
        # merged = {}
        # for c in chan_cols:
        #     means = [s[c]["mean"] for s in stats_list]
        #     stds = [s[c]["std"] for s in stats_list]
        #     merged[c] = {
        #         "mean": float(np.mean(means)),
        #         "std": float(np.mean(stds)) or 1.0,  # approximation across files
        #         "min": float(min(s[c]["min"] for s in stats_list)),
        #         "max": float(max(s[c]["max"] for s in stats_list)),
        #     }
        # return merged

    
    def _apply_normalization(self, df:pd.DataFrame, stats):
            """
            :param df: pd.DataFrame, ema values
            :param stats: dictionary of stats
            :param chan_cols: list of coil columns
            :return out: pd.DataFrame, df post-normalization
            """
            out = df.copy()
            for c in self._chan_cols:
                s = stats[c]
                if self.normalize_method == "zscore":
                    out[c] = (out[c] - s["mean"]) / (s["std"] if s["std"] else 1.0)
                # elif method == "minmax":
                #     span = (s["max"] - s["min"]) or 1.0
                #     out[c] = 2 * (out[c] - s["min"]) / span - 1  # -> [-1, 1]
                elif self.normalize_method == "mean":
                    out[c] = out[c] - s["mean"]
                else:
                    raise ValueError(f"Unknown method: {self.normalize_method}")
            return out

    def _normalize(self):
        if self.normalize_method == 'none':
            self.normalized_data = self.cleaned_data.copy()
        else:
            self.normalized_data = {}
            for name, df in self.cleaned_data.items():
                self.normalized_data[name] = self._apply_normalization(df, self.merged_stats[name])

    def _downsample_ratio(self, df, factor=10):
        """Simple anti-aliased decimation, e.g. 500Hz -> 50Hz."""
        #chan_cols = [c for c in df.columns if c not in ("time", "present")]
        out = {}
        for c in self._chan_cols:
            out[c] = resample_poly(df[c].values, up=1, down=factor)
        n = len(next(iter(out.values())))
        result = pd.DataFrame(out)
        result.insert(0, "time", np.arange(n) / (_DATASET_PARAMS[self.dataset]['sample_rate'] / factor))
        return result
    
    def _downsample_to_wavlm_frames(self, df, num_frames):
        """Interpolate EMA onto the center timestamp of each WavLM output frame."""
        #chan_cols = [c for c in df.columns if c not in _DATASET_PARAMS[self.dataset]['exclude_cols']]
        # Frame i's receptive-field center; using hop*(i+0.5) is the standard
        # approximation used for wav2vec2/HuBERT/WavLM frame-to-time alignment.
        target_times = (np.arange(num_frames) + 0.5) * _WAVLM_HOP
    
        t = df[_DATASET_PARAMS[self.dataset]['time_col']].values
        result = {"time": target_times}
        for c in self._chan_cols:
            f = interp1d(t, df[c].values, kind="linear", bounds_error=False,
                        fill_value=(df[c].values[0], df[c].values[-1]))
            result[c] = f(target_times)
            
        return pd.DataFrame(result)

    def _downsample(self):
        """
        """
        if self.downsample_method == 'none':
            self.downsampled_data = self.normalized_data.copy()
        else:
            self.downsampled_data = {}
            for name, df in self.normalized_data.items():
                if self.downsample_method == 'ratio':
                    self.downsampled_data[name] = self._downsample_ratio(df, self.downsample_factor)
                elif self.downsample_method == 'wavlm':
                    self.downsampled_data[name] = self._downsample_to_wavlm_frames(df, self.downsample_frames[name])

    def correlation(self, to_compare, selected_cols):
        """
        """
        cor = {}
        cors = []
        for n, df in self.downsampled_data.items():
            ema = df[selected_cols].to_numpy()
            compare_ema = to_compare[n].to_numpy()
            n_channels = ema.shape[1]
            r = np.zeros(n_channels)
            p = np.zeros(n_channels)
            for i in range(n_channels):
                r[i], p[i] = pearsonr(ema[:, i], compare_ema[:, i])
            m = r.mean()
            cor[n] = {'r': r, 'p':p, 'mean':m}
            cors.extend(r)
        avg_cor = sum(cors) / len(cors)
        return cor, avg_cor
    
    def _plot_articulatory_trajectory(self, ax, df, base, cmap="viridis", invert_y=False, mark_start_end=True):
        """
        """
        x_col, y_col = self._find_xy_columns(df, base)
        x = df[x_col].values
        y = df[y_col].values
        t = df["time"].values if "time" in df.columns else np.arange(len(x))
    
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
        norm = Normalize(vmin=t.min(), vmax=t.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(t[:-1])
        lc.set_linewidth(2.5)
        line = ax.add_collection(lc)
    
        if mark_start_end:
            ax.scatter(x[0], y[0], color="black", marker="o", s=50, zorder=5, label="start")
            ax.scatter(x[-1], y[-1], color="black", marker="s", s=50, zorder=5, label="end")
    
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(y.min() - 1, y.max() + 1)
        if invert_y:
            ax.invert_yaxis()  # some EMA setups have y increasing downward
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(base)
        ax.legend(loc="upper right", fontsize=8)
        return line
    
    def plot_one_coil(self, df, channel):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        

        last_line = self._plot_articulatory_trajectory(ax, df, channel, cmap="viridis", invert_y=False)
        cbar = fig.colorbar(last_line, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("time (s)")

    def plot_all_coils(self, df):
        bases = sorted(set(c[:-2] for c in df.columns if c.lower().endswith(("_x", "_y"))))
        channels = list(set([b.split("_")[0] for b in bases]))
        n = len(bases)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
        axes = axes[0]

        last_line = None
        for ax, ch in zip(axes, channels):
            last_line = self._plot_articulatory_trajectory(ax, df, ch, cmap="viridis", invert_y=False)
        cbar = fig.colorbar(last_line, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("time (s)")

        #fig.savefig(args.output, dpi=150, bbox_inches="tight")
