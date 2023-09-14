#!/usr/bin/env python3

import os
import glob
import re


def get_frame(start_sec, end_sec, audio_len, min_dur, sr=16000):
    start_fr = min(audio_len, max(0, int(start_sec * sr)))
    end_fr = min(audio_len, int(end_sec * sr))

    need_to_add_fr = int(min_dur * sr) - (end_fr - start_fr)
    if need_to_add_fr > 0:
        start_fr = max(0, start_fr - int(need_to_add_fr / 2))
        end_fr = min(audio_len, end_fr + int(need_to_add_fr / 2))

    # edge case
    need_to_add_fr = int(min_dur * sr) - (end_fr - start_fr)
    if need_to_add_fr > 0:
        if start_fr == 0:
            end_fr = min(audio_len, end_fr + need_to_add_fr)
        elif end_fr == audio_len:
            start_fr = max(0, start_fr - need_to_add_fr)
    return start_fr, end_fr


def get_list(in_wav="", out_tsv="", in_dir="", out_dir="", rank=0, world_size=8):
    """get list of input wav and outpu tsv files"""

    in_wav_list = []
    out_tsv_list = []
    if in_wav != "" and out_tsv != "":
        in_wav_list = [in_wav]
        out_tsv_list = [out_tsv]
    elif in_dir != "" and out_dir != "":
        in_wav_list = sorted(glob.glob(f"{in_dir}/**/*.wav", recursive=True))
        in_wav_list = [x for i, x in enumerate(in_wav_list) if i % world_size == rank]

        for _in_wav in in_wav_list:
            basename = os.path.splitext(re.sub(in_dir, "", _in_wav))[0]
            out_tsv_list.append(os.path.join(out_dir, f"{basename}.tsv"))
    else:
        raise ValueError("in_wav and out_tsv or in_dir and out_dir must be set.")
    return in_wav_list, out_tsv_list


def dump_result(diar_result, out_tsv):
    """dump_result"""

    out_dir = os.path.dirname(out_tsv)
    if not os.path.exists(out_dir) and out_dir != "":
        os.makedirs(out_dir)
    with open(out_tsv, "w", encoding="utf-8") as out_f:
        for res in diar_result:
            out_f.write(f"{res}\n")
