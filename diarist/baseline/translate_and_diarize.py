#!/usr/bin/env python3

import os

import fire
import torch
import torchaudio
import whisper
from speechbrain.pretrained import EncoderClassifier

from diarist.baseline.utils import get_frame, get_list, dump_result
from diarist.baseline.clustering import clustering


def process_one_sample(
    in_wav,
    st_model,
    spk_model,
    beam_size=5,
    condition_on_previous_text=False,
    num_speakers=-1,
    max_num_speakers=6,
    min_dur=0.8,
):
    """process_one_sample"""

    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    # Speech translation
    decode_options = {
        "task": "translate",
        "beam_size": beam_size,
        "condition_on_previous_text": condition_on_previous_text,
    }
    st_result = st_model.transcribe(in_wav, **decode_options)

    # Speaker embedding extraction
    audio, sr = torchaudio.load(in_wav)
    assert sr == 16000
    assert audio.shape[0] == 1

    # extract speaker embedding with minimum duration of [min_dur] sec for each segment
    emb_list = []
    for i, res in enumerate(st_result["segments"]):
        start_fr, end_fr = get_frame(res["start"], res["end"], audio.shape[1], min_dur)
        embedding = spk_model.encode_batch(audio[0, start_fr:end_fr]).reshape(-1)
        emb_list.append(embedding)
    stacked_embedding = torch.stack(emb_list)

    # clustering
    clust_result = clustering(
        stacked_embedding, num_speakers=num_speakers, max_num_speakers=max_num_speakers
    )

    diar_result = []
    for i, res in enumerate(st_result["segments"]):
        start = res["start"]
        end = res["end"]
        text = res["text"]
        text = " ".join(text.split())  # remove redundant spaces
        diar_result.append(f"guest_{clust_result[i]}\t{start}\t{end}\t{text}")
    return diar_result


def translate_and_diarize_main(
    in_wav="",
    out_tsv="",
    in_dir="",
    out_dir="",
    st_model_size="small",
    beam_size=5,
    num_speakers=-1,
    max_num_speakers=6,
    min_dur=0.8,
    rank=0,
    world_size=8,
):
    """main"""
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    # set model
    if torch.cuda.is_available():
        st_model = whisper.load_model(st_model_size, device=f"cuda:{rank}")
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": f"cuda:{rank}"},
        )
    else:
        st_model = whisper.load_model(st_model_size)
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

    # set input and output
    in_wav_list, out_tsv_list = get_list(
        in_wav, out_tsv, in_dir, out_dir, rank, world_size
    )

    # process files
    for _in_wav, _out_tsv in zip(in_wav_list, out_tsv_list):
        if os.path.exists(_out_tsv):
            print(f"{_out_tsv} already exists. Skip.")
            continue

        print(f"Processing {_in_wav}")
        diar_result = process_one_sample(
            _in_wav,
            st_model,
            spk_model,
            beam_size=beam_size,
            num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            min_dur=min_dur,
        )

        print(f"Generate {_out_tsv}")
        dump_result(diar_result, _out_tsv)


def main():
    fire.Fire(translate_and_diarize_main)


if __name__ == "__main__":
    fire.Fire(translate_and_diarize_main)
