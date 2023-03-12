# Instrument-Specific Emotion Recognition

## Usage instructions
1. This repository contains submodules, so it is easiest to clone it recursively: `git clone --recursive https://github.com/jpauwels/instrument_emotion_recognition.git`
2. Install all required libraries: `python -m pip install [--user] -r requirements.txt`.
3. Custom  [`tensorflow_datasets`](https://www.tensorflow.org/datasets) are included, which need to be constructed before running the code. To do so, first place the zipped audio files in `$TFDS_DATA_DIR/downloads/manual` (which defaults to `$HOME/tensorflow_datasets/downloads/manual`). Then run the following Python code from the repository root:

        from instrument_emotion_datasets import acoustic_guitar_emotion_recognition, electric_guitar_emotion_recognition, piano_emotion_recognition
        import tensorflow_datasets as tfds
        tfds.load('acoustic_guitar_emotion_recognition')
        tfds.load('electric_guitar_emotion_recognition')
        tfds.load('piano_emotion_recognition')

    and wait for the process to complete.
4. The emotion recognising neural networks can be trained using a CLI. For a full list of options, run `python instrument_emotion_recognition.py -h`. All options (except the paths) can be specified multiple times, which will cause the training to run multiple times with all possible combinations of the given options (and the default values for missing options). To approximately replicate the results of the [TASLP paper](https://doi.org/10.1109/TASLP.2021.3138709), the following two-stage process should be run:

        python instrument_emotion_recognition.py -i acoustic_guitar -w MTT_musicnn -l 0.001 --finetuning no --save-model-dir 'saved-models'
        python instrument_emotion_recognition.py -i acoustic_guitar -w 'saved-models/instrument_emotion_recognition/<id of run above>' -l 0.001 --finetuning yes --save-model-dir 'saved-models'
