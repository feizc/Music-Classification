# fine-tunning in the down-stream music classification task  
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchaudio 
from tqdm import tqdm 
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AdamW, get_linear_schedule_with_warmup
from model import Wav2Vec2ForSpeechClassification
import json 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class Trainconfig: 
    wav2vec_model_path = './ckpt/wav2vec' 
    ckpt_path = './ckpt/classifier'
    data_path = './data/meituan/train.json' 
    lr = 2e-5
    warmup_steps = 5000 
    batch_size = 1
    epochs = 1




def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


class MusicDataset(Dataset): 
    def __init__(self, data_path, feature_extractor): 
        self.feature_extractor = feature_extractor 
        with open(data_path, 'r', encoding='utf-8') as f: 
            self.data_list = json.load(f) 
        self.sampling_rate = feature_extractor.sampling_rate

    def __len__(self): 
        return len(self.data_list) 
    
    def __getitem__(self, item): 
        music_path = self.data_list[item]['audio_path'] 
        speech = speech_file_to_array_fn(music_path, self.sampling_rate) 
        if len(speech.shape) > 1: 
            speech = speech[0, :]
        inputs = self.feature_extractor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True) 
        inputs['input_values'] = inputs['input_values'].to(device)
        label = torch.Tensor([self.data_list[item]['label']]).long()
        return inputs, label 



def train(dataset, model, args): 
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    )
    acc = .0 
    for epoch in range(args.epochs): 
        progress = tqdm(total=len(train_dataloader), desc='music transformer')
        for idx, (inputs, label) in enumerate(train_dataloader): 
            model.zero_grad()
            label = label.to(device) 
            inputs['input_values'] = inputs['input_values'].squeeze(0)
            outputs = model(**inputs).logits[0]
            predict_y = torch.max(outputs, 0)
            acc += (predict_y == label).sum().item() / predict_y.size(0) 
            loss = F.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), label.flatten(), ignore_index=-1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item(), 'acc': acc / (idx + 1)})
            progress.update()

            break 
        progress.close()



def main(): 
    args = Trainconfig() 
    model_config = AutoConfig.from_pretrained(args.wav2vec_model_path) 
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_path)
    model =  Wav2Vec2ForSpeechClassification.from_pretrained(args.wav2vec_model_path).to(device) 
    
    # adjust the output dimension and fixed the pre feature 
    model.classifier.out_proj = nn.Linear(768, 20) 
    for n, p in model.named_parameters():
        if 'classifier' not in n: 
            p.requires_grad = False 
    model = model.to(device)
    dataset = MusicDataset(args.data_path, feature_extractor) 
    train(dataset, model, args)



if __name__ == '__main__':
    main()
