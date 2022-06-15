import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

from model import Wav2Vec2ForSpeechClassification

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "./ckpt"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model =  Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


id2label = {
    "0": "忧郁",
    "1": "古典",
    "2": "乡村",
    "3": "迪斯科",
    "4": "嘻哈",
    "5": "爵士",
    "6": "金属",
    "7": "爆裂",
    "8": "节奏",
    "9": "摇滚"
  }



def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs

# dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification 

path = "./data/gtzan/genres_original/disco/disco.00067.wav"
outputs = predict(path, sampling_rate)
print(outputs)
