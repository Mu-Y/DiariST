#!/usr/bin/env python3

import sys
import json


if __name__ == "__main__":
    in_tsv = sys.argv[1]
    in_json = sys.argv[2]

    translation_data = {}
    with open(in_tsv, "r") as f:
        for line in f:
            line = line.strip()
            session, speaker, start, end, text, translation = line.split("\t")
            if session not in translation_data:
                translation_data[session] = []
            translation_data[session].append(
                {
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker,
                    "text": text,
                    "translation": translation,
                }
            )

    with open(in_json, "r") as f:
        data = json.load(f)

        for mini_session in data:
            start = mini_session["start"]
            end = mini_session["end"]
            session = mini_session["session"]
            mini_session["data"] = []
            for elem in translation_data[session]:
                elem_start = elem["start"]
                elem_end = elem["end"]
                if start <= elem_start and elem_end <= end:
                    mini_session["data"].append(elem)
            mini_session["data"] = sorted(
                mini_session["data"], key=lambda x: x["start"]
            )
        print(json.dumps(data, indent=4, ensure_ascii=False))
