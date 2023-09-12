#!/usr/bin/env python3

import sys
import os
import json
import glob
import re

import soundfile as sf
import numpy as np

ROOT_DIR = "data/DiariST-AliMeeting/"
SUBSETS = ["dev", "test"]


def fix_timing(data, start):
    """fix_timing"""
    for elem in data:
        elem["start"] -= start
        elem["end"] -= start
    return data


def gen_sdm(in_json, ali_meeting_dir, subset):
    """generate single distant microphone data"""
    with open(in_json) as fp:
        all_data = json.load(fp)

    prev_session = ""
    for one_session in all_data:
        session = one_session["session"]
        data = one_session["data"]
        start = one_session["start"]
        end = one_session["end"]

        if prev_session != session:
            if subset == "dev":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Eval_Ali/Eval_Ali_far/audio_dir/{session}_*.wav"
                )
            elif subset == "test":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Test_Ali/Test_Ali_far/audio_dir/{session}_*.wav"
                )
            assert len(audio_files) == 1
            audio_file = audio_files[0]

            audio, sr = sf.read(audio_file)
        prev_session = session

        out_wav = f"{ROOT_DIR}/SDM/{subset}/{session}-{start}-{end}.wav"
        out_json = f"{ROOT_DIR}/SDM/{subset}/{session}-{start}-{end}.json"
        mkdir = os.path.dirname(out_wav)
        if not os.path.exists(mkdir):
            os.makedirs(mkdir, exist_ok=True)

        start_fr = int(start * sr)
        end_fr = int(end * sr)

        if os.path.exists(out_wav):
            print(f"{out_wav} exists, skip")
        else:
            sf.write(out_wav, audio[start_fr:end_fr, 0], sr)

        time_fixed_data = fix_timing(data, start)
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(time_fixed_data, fp, indent=4, ensure_ascii=False)


def gen_ihm_mix(in_json, ali_meeting_dir, subset):
    """generate mixture of independent head microphones"""
    with open(in_json) as fp:
        all_data = json.load(fp)

    prev_session = ""
    max_dur_idx = 0
    sr = 16000
    for one_session in all_data:
        session = one_session["session"]
        data = one_session["data"]
        start = one_session["start"]
        end = one_session["end"]

        # create IHM-MIX
        if prev_session != session:
            if subset == "dev":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Eval_Ali/Eval_Ali_near/audio_dir/{session}_*.wav"
                )
            elif subset == "test":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Test_Ali/Test_Ali_near/audio_dir/{session}_*.wav"
                )
            assert len(audio_files) <= 4

            audios = []
            max_dur = 0
            max_dur_idx = 0
            for i, audio_file in enumerate(audio_files):
                audio, sr = sf.read(audio_file)
                audios.append(audio)
                if len(audio) > max_dur:
                    max_dur = len(audio)
                    max_dur_idx = i
            for i in range(len(audios)):
                if i != max_dur_idx:
                    audios[max_dur_idx][: len(audios[i])] += audios[i]
        prev_session = session

        out_wav = f"{ROOT_DIR}/IHM-MIX/{subset}/{session}-{start}-{end}.wav"
        out_json = f"{ROOT_DIR}/IHM-MIX/{subset}/{session}-{start}-{end}.json"
        mkdir = os.path.dirname(out_wav)
        if not os.path.exists(mkdir):
            os.makedirs(mkdir, exist_ok=True)

        start_fr = int(start * sr)
        end_fr = int(end * sr)

        if os.path.exists(out_wav):
            print(f"{out_wav} exists, skip")
        else:
            sf.write(out_wav, audios[max_dur_idx][start_fr:end_fr], sr)

        time_fixed_data = fix_timing(data, start)
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(time_fixed_data, fp, indent=4, ensure_ascii=False)


def gen_ihm_cat(in_json, ali_meeting_dir, subset):
    """generate concat of independent head microphones"""
    with open(in_json) as fp:
        all_data = json.load(fp)

    prev_session = ""
    for one_session in all_data:
        session = one_session["session"]
        data = one_session["data"]
        session_start = one_session["start"]
        session_end = one_session["end"]

        if prev_session != session:
            if subset == "dev":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Eval_Ali/Eval_Ali_near/audio_dir/{session}_*.wav"
                )
            elif subset == "test":
                audio_files = glob.glob(
                    f"{ali_meeting_dir}/Test_Ali/Test_Ali_near/audio_dir/{session}_*.wav"
                )
            assert len(audio_files) <= 4

            sr = 16000
            audios = {}
            for i, audio_file in enumerate(audio_files):
                basename = os.path.basename(audio_file)
                match = re.search(r"\w+_(N_SPK\d+).wav", basename)
                if not match:
                    raise RuntimeError(f"Failed to parse {basename}")
                spk = match.group(1)
                audio, sr = sf.read(audio_file)
                assert sr == 16000
                audios[spk] = audio

        audio_to_concate = []
        for i, elem in enumerate(data):
            spk = elem["speaker"]
            start_fr = int(elem["start"] * sr)
            end_fr = int(elem["end"] * sr)
            audio_to_concate.append(audios[spk][start_fr:end_fr])
        audio = np.concatenate(audio_to_concate)

        new_data = []
        new_start = 0
        new_end = 0
        for elem in data:
            start = elem["start"]
            end = elem["end"]
            new_start = new_end
            new_end = new_start + end - start
            new_data.append(
                {
                    "speaker": elem["speaker"],
                    "start": new_start,
                    "end": new_end,
                    "text": elem["text"],
                    "translation": elem["translation"],
                }
            )

        out_wav = (
            f"{ROOT_DIR}/IHM-CAT/{subset}/{session}-{session_start}-{session_end}.wav"
        )
        out_json = (
            f"{ROOT_DIR}/IHM-CAT/{subset}/{session}-{session_start}-{session_end}.json"
        )
        mkdir = os.path.dirname(out_wav)
        if not os.path.exists(mkdir):
            os.makedirs(mkdir, exist_ok=True)

        if os.path.exists(out_wav):
            print(f"{out_wav} exists, skip")
        else:
            sf.write(out_wav, audio, sr)

        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(new_data, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ali_meeting_dir>")
        sys.exit(1)

    ALI_MEETING_DIR = sys.argv[1]

    for subset in SUBSETS:
        json_file = f"{ROOT_DIR}/{subset}.json"

        gen_sdm(json_file, ALI_MEETING_DIR, subset)
        gen_ihm_mix(json_file, ALI_MEETING_DIR, subset)
        gen_ihm_cat(json_file, ALI_MEETING_DIR, subset)
