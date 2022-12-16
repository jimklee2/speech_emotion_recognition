# Speech Emotion Recognition
___
# How to use
- ## How to trained (pytorch)
  - First, clone this repo.
  - Train code created by jupyter notebook
  - Trainer code located in main folder(train.ipynb)
  - Locate your wav files in your new folder
  - Select model in torchvision.models(using DenseNet in this code)

- ## How to record
  - run record.py -> record your voice for 4 seconds -> .wav file will saved in './predict_audio'
    ```
    python3 record.py
    ```

- ## How to predict
  - Locate wav file in './predict_audio'.
  - You can use the file you recorded, or another file

    ```
    python predict.py
    ```

