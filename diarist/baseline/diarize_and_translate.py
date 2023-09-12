#!/usr/bin/env python3

import os

import fire
import torch
import torchaudio
import whisper
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import VAD


from diarist.baseline.utils import get_frame, get_list, dump_result
from diarist.baseline.clustering import clustering


def process_one_sample(
    in_wav,
    st_model,
    spk_model,
    vad_model=None,
    beam_size=5,
    condition_on_previous_text=False,
    num_speakers=-1,
    max_num_speakers=6,
    window_size=1.2,
    window_shift=0.6,
):
    """process_one_sample"""

    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    #
    # Speaker embedding extraction
    #
    audio, sr = torchaudio.load(in_wav)
    assert sr == 16000
    assert audio.shape[0] == 1

    # VAD
    if vad_model is not None:
        boundaries = vad_model.get_speech_segments(in_wav).tolist()
    else:
        boundaries = [[0.0, audio.shape[1] / sr]]

    # extract speaker embeddings with sliding window
    emb_list = []
    for boundary_start, boundary_end in boundaries:
        boundary_dur = boundary_end - boundary_start
        num_shift = int(boundary_dur / window_shift)
        for i in range(num_shift):
            start_sec = boundary_start + window_shift * i
            start_fr, end_fr = get_frame(
                start_sec, start_sec + window_size, audio.shape[1], window_size
            )
            embedding = spk_model.encode_batch(audio[0, start_fr:end_fr]).reshape(-1)
            emb_list.append(embedding)
    stacked_embedding = torch.stack(emb_list)

    # clustering
    clust_result = clustering(
        stacked_embedding, num_speakers=num_speakers, max_num_speakers=max_num_speakers
    )

    # aggregate segments for the same speaker
    segment_result = []
    embedding_index = 0
    for boundary_start, boundary_end in boundaries:
        seg_start = boundary_start
        prev_time = boundary_start
        prev_spk = -1
        boundary_dur = boundary_end - boundary_start
        num_shift = int(boundary_dur / window_shift)
        for i in range(num_shift):
            # for i in range(len(emb_list)):
            cur_time = min(
                boundary_start + window_size * 0.5 + window_shift * (i + 0.5),
                boundary_end,
            )
            cur_spk = clust_result[embedding_index].item()
            embedding_index += (
                1  # this must be align with the embedding extraction loop
            )

            if cur_spk != prev_spk and prev_spk != -1:
                segment_result.append([seg_start, prev_time, prev_spk])
                seg_start = cur_time
            prev_time = cur_time
            prev_spk = cur_spk
        if seg_start != prev_time:
            segment_result.append([seg_start, prev_time, prev_spk])

    #
    # apply speech translation
    #
    diar_result = []
    for start, end, spk in segment_result:
        decode_options = {
            "task": "translate",
            "beam_size": beam_size,
            "condition_on_previous_text": condition_on_previous_text,
        }
        st_result = st_model.transcribe(
            audio[0, int(start * sr) : int(end * sr)], **decode_options
        )
        text = st_result["text"]
        text = " ".join(text.split())  # remove redundant spaces
        if text != "":
            diar_result.append(f"guest_{spk}\t{start}\t{end}\t{text}")
    return diar_result


def diarize_and_translate_main(
    in_wav="",
    out_tsv="",
    in_dir="",
    out_dir="",
    st_model_size="small",
    beam_size=5,
    num_speakers=-1,
    max_num_speakers=6,
    window_size=1.2,
    window_shift=0.6,
    apply_VAD=True,
    rank=0,
    world_size=8,
):
    """main"""

    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    # set models
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
    vad_model = None
    if apply_VAD:
        vad_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")

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
            vad_model=vad_model,
            beam_size=beam_size,
            num_speakers=num_speakers,
            max_num_speakers=max_num_speakers,
            window_size=window_size,
            window_shift=window_shift,
        )

        print(f"Generate {_out_tsv}")
        dump_result(diar_result, _out_tsv)


def main():
    fire.Fire(diarize_and_translate_main)


if __name__ == "__main__":
    fire.Fire(diarize_and_translate_main)
