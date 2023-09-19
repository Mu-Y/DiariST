# DiariST
This repository maintains the data and code used for the paper "DiariST: Streaming Speech Translation with Speaker Diarization". 

[Read the paper](https://arxiv.org/abs/2309.08007)
## Overview
End-to-end speech translation (ST) for conversation recordings involves several underexplored challenges such as speaker diarization (SD) without accurate word time stamps and handling of overlapping speech in a streaming fashion. Due to the absence of evaluation benchmarks in this area, we develop a new evaluation dataset, **DiariST-AliMeeting**, by translating the reference Chinese transcriptions of the [AliMeeting](https://www.openslr.org/119/) into English. We also propose new metrics, called **Speaker-Agnostic BLEU (SAgBLEU)** and **Speaker-Attributed BLEU (SAtBLEU)**, to measure the ST quality while taking SD accuracy into account. In our paper [DiariST: Streaming Speech Translation with Speaker Diarization](https://arxiv.org/abs/2309.08007), we further propose the first streaming ST and SD system, named **DiariST**, by integrating [token-level serialized output training](https://arxiv.org/abs/2202.00842) and [t-vector](https://arxiv.org/abs/2203.16685) into [a neural transducer-based streaming ST system](https://arxiv.org/abs/2204.05352). To facilitate the research in this new direction, we release the evaluation data, the offline baseline systems, and the evaluation code, used in the paper.


## Prerequisites
- Linux
  - python 3.9

## Installation
```sh
pip install git+https://github.com/openai/whisper.git
git clone https://github.com/Mu-Y/DiariST.git
cd DiariST
pip install -e .
```

## How to generate the Diarist-AliMeeting data
```sh
$ ./run_prepare_data.sh
```
The audio files and corresponding reference json files are generated under ./data/ directory as following structures.
```
data
└── DiariST-AliMeeting
     └── [IHM-CAT, IHM-MIX, SDM]
         └── [dev, test]
             ├── Rxxx_Myyy_start_end.wav
             ├── Rxxx_Myyy_start_end.json
             ├── ...
```

## How to run the baseline system
- Translation --> Diarization
  - Option 1: Run "translation --> diarization" baseline for one audio sample.
  ```sh
  $ diarist_baseline_td --in_wav data/DiariST-AliMeeting/IHM-CAT/test/R8002_M8002-0-249.06.wav --out_tsv result/DiariST-AliMeeting/IHM-CAT/test/R8002_M8002-0-249.06.tsv
  ```

  - Option 2: Run "translation --> diarization" baseline for all audio samples under data/DiariST-AliMeeting/IHM-CAT/test/. (CAUTION: it will take long time because this script applies the baseline for each audio one by one without any parallelization.)
  ```sh
  $ diarist_baseline_td --in_dir data/DiariST-AliMeeting/IHM-CAT/test/ --out_dir result/DiariST-AliMeeting/IHM-CAT/test/
  ```

- Diarization --> Translation
  - Please use a command "diarist_baseline_dt" instead of "diarist_baseline_td"

## How to evaluate the result
Assuming that reference translations are stored under "./data/DiariST-AliMeeting/IHM-CAT/test/" and the diarized speech translation results are stored under "./result/DiariST-AliMeeting/IHM-CAT/test/" in TSV format, you can compute SAgBLEU and SAtBLEU using the following command.
```sh
$ diarist_eval \
    --ref_dir ./data/DiariST-AliMeeting/IHM-CAT/test/ \
    --hyp_dir ./result/DiariST-AliMeeting/IHM-CAT/test/
```

It will compute SAgBLEU and SAtBLEU score as follows. (Note: Our results in the paper were obtained using the Tesla V100 with 16GB of memory. The results may vary depending on the computational environment.)
```sh
Found 195 files in result/DiariST-AliMeeting/IHM-CAT/test/
SAgBLEU: 18.45
SAtBLEU: 16.81
```

Note that SAgBLEU and SAtBLEU are uttearnce-order sensitive, but not time-stamp sensitive. If your speech translation system does not generate precise timestamps, you can simply set dummy timestamps in the TSV file.

## License
|  | License |
| ------------- |:-------------:|
| DiariST-AliMeeting | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Anything else | [MIT Licence](LICENSE.txt) |

## Citation
Please cite the first paper (yang2023diarist) when you use the code in this repository. If you use the DiariST-AliMeeting test set under data/ directory, please also cite the two papers for AliMeeting corpus (Yu2022M2Met, Yu2022Summary) in addition to the first paper (yang2023diarist).
```
@article{yang2023diarist,
  title={{DiariST}: Streaming Speech Translation with Speaker Diarization},
  author={Yang, Mu and Kanda, Naoyuki and Wang, Xiaofei and Chen, Junkun and Wang, Peidong and Xue, Jian and Li, Jinyu and Yoshioka, Takuya},
  journal={arXiv preprint arXiv:2309.08007},
  year={2023}
}

@inproceedings{Yu2022M2MeT,
  title={M2{M}e{T}: The {ICASSP} 2022 Multi-Channel Multi-Party Meeting Transcription Challenge},
  author={Yu, Fan and Zhang, Shiliang and Fu, Yihui and Xie, Lei and Zheng, Siqi and Du, Zhihao and Huang, Weilong and Guo, Pengcheng and Yan, Zhijie and Ma, Bin and Xu, Xin and Bu, Hui},
  booktitle={Proc. ICASSP},
  pages={6167--6171},
  year={2022},
  organization={IEEE}
}

@inproceedings{Yu2022Summary,
  title={Summary On The {ICASSP} 2022 Multi-Channel Multi-Party Meeting Transcription Grand Challenge},
  author={Yu, Fan and Zhang, Shiliang and Guo, Pengcheng and Fu, Yihui and Du, Zhihao and Zheng, Siqi and Huang, Weilong and Xie, Lei  and Tan, Zheng-Hua and Wang, DeLiang and Qian, Yanmin and Lee, Kong Aik and Yan, Zhijie and Ma, Bin and Xu, Xin and Bu, Hui},
  booktitle={Proc. ICASSP},
  pages={9156--9160},
  year={2022},
  organization={IEEE}
}
```
