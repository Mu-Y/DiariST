from setuptools import setup, find_packages

setup(
    name="diarist",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchaudio",
        "Cython",
        "nemo_toolkit[all]",
        "speechbrain",
        "fire",
        "sacrebleu==2.3.1",
    ],
    entry_points={
        "console_scripts": [
            "diarist_baseline_td=diarist.baseline.translate_and_diarize:main",
            "diarist_baseline_dt=diarist.baseline.diarize_and_translate:main",
            "diarist_eval=diarist.scoring.eval:main"
        ]
    },
)
