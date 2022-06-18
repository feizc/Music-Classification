import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import os 
from tqdm import tqdm 
import json 

from model import Wav2Vec2ForSpeechClassification

import librosa
import numpy as np
import pandas as pd

from data_preprocess import data_preprocess, url_to_wav_file


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name_or_path = "./ckpt/wav2vec"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model =  Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


id2label = {
    0: "忧郁",
    1: "古典",
    2: "乡村",
    3: "迪斯科",
    4: "嘻哈",
    5: "爵士",
    6: "金属",
    7: "爆裂",
    8: "节奏",
    9: "摇滚"
  }



def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate=16000): 
    speech = speech_file_to_array_fn(path, sampling_rate)
    if len(speech.shape) > 1: 
        speech = speech[0, :]
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs['input_values'] = inputs['input_values'].to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0] 
    index = np.argmax(scores)
    # outputs = [{"Label": id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return id2label[index] + '\t' + f"{round(scores[index] * 100, 3):.1f}%" + '\n'

# dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification 

if __name__ == '__main__': 

    excel_path = './data/meituan/music.xlsx' 
    output_path = './data/meituan'
    wav_path = './data/meituan/wav_music'
    url_list, label_list = data_preprocess(excel_path, output_path)  
    result_inference = True # if generate text label 

    if result_inference == True: 
        lines = []
        for url in tqdm(url_list): 
            wav_file = url_to_wav_file(url) 
            wav_file_path = os.path.join(wav_path, wav_file)
    
            outputs = predict(wav_file_path, sampling_rate) 
            print(outputs)
            lines.append(outputs) 
            break

        with open('predict.txt', 'w', encoding='utf-8') as f: 
            for line in lines: 
                f.write(line) 
    else: 
        train_json_path = './data/meituan/train_json'
