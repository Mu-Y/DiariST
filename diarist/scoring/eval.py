#!/usr/bin/env python3

import os
import json
import itertools
from collections import defaultdict

from sacrebleu.metrics import BLEU
import numpy as np
import fire


def sagbleu(preds, ref_json, delimiter="\t"):
    preds = [" ".join(d.split(delimiter)[3:]) for d in preds]
    refs = [d["translation"] for d in ref_json]
    pred = " ".join(preds)  # form a long sequence
    ref = " ".join(refs)

    bleu = BLEU(effective_order=True)
    session_bleu = bleu.corpus_score([pred], [[ref]]).score
    return [pred], [ref], session_bleu


def satbleu(preds, ref_json, delimiter="\t"):
    spk2chunk_ref = defaultdict(list)
    spk2chunk_hyp = defaultdict(list)
    for seg in ref_json:
        seg_spk = seg["speaker"]
        spk2chunk_ref[seg_spk].append(seg["translation"])
    for chunk in preds:
        spk = chunk.split(delimiter)[0]
        hyp = " ".join(chunk.split(delimiter)[3:])
        spk2chunk_hyp[spk].append(hyp)
    bleu = BLEU()

    ref_spks = list(spk2chunk_ref.keys())
    hyp_spks = list(spk2chunk_hyp.keys())

    list_len = max(len(ref_spks), len(hyp_spks))
    if list_len > 6:
        # return None
        print(
            "Num of predicted speakers is larger than 6, evaluating this sample will be slow. "
        )
    perms = list(itertools.permutations(range(list_len)))
    # corpus_bleus = [0] * len(perms)
    if len(ref_spks) < len(hyp_spks):
        ref_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_ref.items()] + [
            ""
        ] * (len(hyp_spks) - len(ref_spks))
        hyp_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_hyp.items()]

    elif len(ref_spks) > len(hyp_spks):
        ref_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_ref.items()]
        hyp_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_hyp.items()] + [
            ""
        ] * (len(ref_spks) - len(hyp_spks))

    elif len(ref_spks) == len(hyp_spks):
        ref_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_ref.items()]
        hyp_texts = [" ".join(spk_txt) for spk, spk_txt in spk2chunk_hyp.items()]

    max_perm = perms[0]
    max_score = bleu.corpus_score([hyp_texts[j] for j in max_perm], [ref_texts]).score

    for i, perm in enumerate(perms[1:]):
        score = bleu.corpus_score([hyp_texts[j] for j in perm], [ref_texts]).score
        if score > max_score:
            max_score = score
            max_perm = perm
    return [hyp_texts[j] for j in max_perm], ref_texts, max_score


def evaluate(ref_dir, hyp_dir):
    full_path_list = [
        os.path.join(hyp_dir, f) for f in os.listdir(hyp_dir) if f.endswith(".tsv")
    ]

    for eval_method in ["SAgBLEU", "SAtBLEU"]:
        dir_scores = {
            "avg_sentence_bleu": 0.0,
            "avg_semi_corpus_bleu": 0.0,
            "details": {},
        }

        total_hyps, total_refs = [], []
        for dname in full_path_list:
            # assume parallel dir structure, search for ref file in `in_dir`

            just_name = os.path.splitext(os.path.basename(dname))[0]
            ref_json_path = os.path.join(ref_dir, f"{just_name}.json")

            with open(ref_json_path, "r") as f:
                ref_json = json.load(f)

            with open(dname, "r") as f:
                preds = f.read().split("\n")

            preds = [p for p in preds if p not in [" ", ""]]
            if eval_method == "SAtBLEU":
                hyp, ref, session_bleu = satbleu(preds, ref_json)
            elif eval_method == "SAgBLEU":
                hyp, ref, session_bleu = sagbleu(preds, ref_json)
            else:
                raise NotImplementedError
            total_hyps.extend(hyp)
            total_refs.extend(ref)

            dir_scores["details"].update({dname: {"session_bleu": session_bleu,}})
        avg_session_bleu = np.mean(
            [d["session_bleu"] for d in dir_scores["details"].values()]
        )
        dir_scores["avg_session_bleu"] = avg_session_bleu

        bleu = BLEU()
        lang_level_corpus_bleu = bleu.corpus_score(total_hyps, [total_refs]).score
        dir_scores["corpus_bleu"] = lang_level_corpus_bleu

        print("{}: {:.2f}".format(eval_method, dir_scores["corpus_bleu"]))


def main():
    fire.Fire(evaluate)


if __name__ == "__main__":
    fire.Fire(evaluate)
