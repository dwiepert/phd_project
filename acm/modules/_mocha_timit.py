import struct
import sys
from pathlib import Path
from typing import Union
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import json 
import random

from scipy.stats import pearsonr

class MochaTimit():
    """
    Helper functions for Mocha-timit
    """
    def load_ema(self, path:Union[str, Path]):
        """
        Read .ema file

        :param path: path-like, full file path for single .ema file
        :return df: pd.DataFrame with file values
        :return meta: dict, metadata for file
        """
        with open(path, "rb") as f:
                raw = f.read()
        
        # --- split header / body -------------------------------------------------
        header_end_marker = b"EST_Header_End\n"
        idx = raw.find(header_end_marker)
        if idx == -1:
            # some files use \r\n
            header_end_marker = b"EST_Header_End\r\n"
            idx = raw.find(header_end_marker)
        if idx == -1:
            raise ValueError("Could not find 'EST_Header_End' — not a valid EST track file?")
    
        header_bytes = raw[:idx]
        body = raw[idx + len(header_end_marker):]
    
        header_text = header_bytes.decode("ascii", errors="replace")
        lines = [l.strip() for l in header_text.splitlines() if l.strip()]
    
        num_frames = None
        num_channels = None
        byte_order = "01"  # 01 = little-endian in EST convention; 10 = big-endian
        data_type = "binary"
        channel_names = {}
    
        for line in lines:
            parts = line.split(None, 1)
            if not parts:
                continue
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ""
    
            if key == "NumFrames":
                num_frames = int(val)
            elif key == "NumChannels":
                num_channels = int(val)
            elif key == "ByteOrder":
                byte_order = val.strip()
            elif key == "DataType":
                data_type = val.strip()
            elif key.startswith("Channel_"):
                try:
                    ch_idx = int(key.split("_", 1)[1])
                    channel_names[ch_idx] = val.strip()
                except ValueError:
                    pass
    
        if num_frames is None or num_channels is None:
            raise ValueError("Header missing NumFrames/NumChannels — dumping header for inspection:\n" + header_text)
    
        endian = "<" if byte_order in ("01", "littleendian", "little") else ">"
        floats_per_frame = 2 + num_channels  # time, present-flag, then channel values
        fmt = f"{endian}{floats_per_frame}f"
        frame_size = struct.calcsize(fmt)
    
        expected_bytes = frame_size * num_frames
        if len(body) < expected_bytes:
            print(
                f"Warning: body has {len(body)} bytes, expected {expected_bytes} "
                f"for {num_frames} frames x {floats_per_frame} floats. "
                "File may be truncated or ASCII-typed.",
                file=sys.stderr,
            )
            num_frames = len(body) // frame_size
    
        rows = []
        for i in range(num_frames):
            chunk = body[i * frame_size:(i + 1) * frame_size]
            if len(chunk) < frame_size:
                break
            values = struct.unpack(fmt, chunk)
            rows.append(values)
    
        columns = ["time", "present"]

        for c in range(num_channels):
            ch_n = channel_names.get(c, f"ch{c}")
            if ch_n == '****':
                ch_n = f'rm_{c}'
            columns.append(ch_n)
    
        df = pd.DataFrame(rows, columns=columns)
        cols_to_drop = df.columns[df.columns.str.contains('rm')]
        df = df.drop(cols_to_drop, axis=1)

        meta = {
            "num_frames": num_frames,
            "num_channels": num_channels,
            "byte_order": byte_order,
            "data_type": data_type,
            "channel_names": channel_names,
        }

        #return df, meta
        return df

    def _interpolate_dropouts(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Mark present==0 frames as NaN and linearly interpolate each channel.
        :param df: pd.DataFrame, ema values
        :return out: pd.DataFrame, df post interpolation
        """
        chan_cols = [c for c in df.columns if c not in ("time", "present")]
        out = df.copy()
        dropout = out["present"] != 1
        out.loc[dropout, chan_cols] = np.nan
        out[chan_cols] = out[chan_cols].interpolate(method="linear", limit_direction="both")
        return out

    def _head_movement_correct(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Subtract reference_x / reference_y from every other coil's x/y columns.
        :param df: pd.DataFrame, ema values
        :return out: pd.DataFrame, df post-correction
        """
        rx, ry = f"bn_x", f"bn_y"
        if rx not in df.columns or ry not in df.columns:
            raise ValueError(
                f"Reference channels '{rx}'/'{ry}' not found. "
                f"Available: {[c for c in df.columns if c not in ('time', 'present')]}"
            )
        out = df.copy()
        for c in out.columns:
            if c in ("time", "present", rx, ry):
                continue
            if c.endswith("_x"):
                out[c] = out[c] - out[rx]
            elif c.endswith("_y"):
                out[c] = out[c] - out[ry]
        return out

    def _find_xy_columns(self, df, base):
        base = base.lower()
        cols = {c.lower(): c for c in df.columns}
        x_col = cols.get(f"{base}_x")
        y_col = cols.get(f"{base}_y")
        if x_col is None or y_col is None:
            raise ValueError(
                f"Couldn't find '{base}_x'/'{base}_y' columns. "
                f"Available channels: {sorted(set(c[:-2] for c in df.columns if c.lower().endswith(('_x', '_y'))))}"
            )
        return x_col, y_col

    def get_name(self, path):
        return path.stem 
        
    def get_speaker(self, path):
        return path.stem.split("_")[0]

    def get_session(self, path):
        return str(random.randint(0, 10))

    def clean_ema(self, df):
        df = self._interpolate_dropouts(df)
        df = self._head_movement_correct(df)
        return df