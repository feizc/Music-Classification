import pandas as pd
import os 
import requests 
from pydub import AudioSegment



def read_excel_data(excel_path): 
    df = pd.read_excel(excel_path, engine='openpyxl')
    url_list = df['content'].values.tolist() 
    label_list = df['category'].values.tolist() 
    return url_list, label_list 


# convert mp3 to wav formation 
def mp3_to_wav(mp3_path, wav_path): 
    mp3_file_list = os.listdir(mp3_path) 
    for mp3_name in mp3_file_list: 
        src_path = os.path.join(mp3_path, mp3_name) 
        tgt_path = os.path.join(wav_path, mp3_name.split('.')[0] + '.wav') 
        sound = AudioSegment.from_mp3(src_path)
        sound.export(tgt_path, format="wav")



def url_to_wav_file(url): 
    return url.split('/')[-1].split('.')[0] + '.wav'


# generated url-label list 
def data_preprocess(excel_path, output_path, down_load=False): 
    url_list, label_list = read_excel_data(excel_path)  
    mp3_path = os.path.join(output_path, 'original_music')
    wav_path = os.path.join(output_path, 'wav_music')
    if down_load == True:
        for url in url_list: 
            audio = requests.get(url, stream=True) 
            with open(os.path.join(output_path, url.split('/')[-1]), 'wb') as f: 
                for chunk in audio.iter_content():
                    f.write(chunk) 
            print(url) 
    
        mp3_to_wav(mp3_path, wav_path)  
    return url_list, label_list 


# project label id to text information 
def label2text_dict_generate(excel_path): 
    df = pd.read_excel(excel_path, engine='openpyxl') 
    id_list = [] 
    text_list = []
    id_list += df['cat0_id'].values.tolist() 
    text_list += df['cat0_name'].values.tolist() 
    id_list += df['cat1_id'].values.tolist() 
    text_list += df['cat1_name'].values.tolist() 
    id_list += df['cat2_id'].values.tolist() 
    text_list += df['cat2_name'].values.tolist() 
    print(len(id_list), len(text_list)) 

    label2text_dict = dict() 
    for i in range(len(id_list)): 
        if id_list[i] == 0:
            continue 
        label2text_dict[str(id_list[i])] = text_list[i]
    return label2text_dict 


def add_text_excel(label_list, label2text_dict): 
    lines = [] 
    for label in label_list: 
        line = '' 
        t_label_list = label.split(',')
        for t_label in t_label_list: 
            if t_label in label2text_dict.keys(): 
                line += label2text_dict[t_label] + ',' 
        line = line[:-1] + '\n'
        lines.append(line)
    
    with open('label.txt', 'w', encoding='utf-8') as f: 
        for line in lines: 
            f.write(line)


# transfor multi-label to single label  label_num = 908 
def multi_to_single(url_list, label_list, label2text_dict, wav_path): 
    label2num_dict = dict() 
    num2label_dict = dict() 

    i = 0 
    for label in label2text_dict.keys(): 
        label2num_dict[label] = i 
        num2label_dict[i] = label 
        i += 1 
    
    data = [] 
    for i in range(len(url_list)): 
        file_name = url_to_wav_file(url_list[i]) 
        file_path = os.path.join(wav_path, file_name) 
        t_label_list = label_list[i].split(',') 
        for label in t_label_list: 
            if label not in label2num_dict.keys():
                continue
            item = dict() 
            item['audio_path'] = file_path 
            item['label'] = label2num_dict[label] 
            data.append(item) 
    print('data size: ', len(data))
    return data






if __name__ == '__main__': 
    excel_path = './data/meituan/music.xlsx' 
    category_excel_path = './data/meituan/category.xlsx'
    output_path = './data/meituan'
    wav_path = './data/meituan/wav_music' 

    url_list, label_list = data_preprocess(excel_path, output_path)  
    
    label2text_dict = label2text_dict_generate(category_excel_path) 

    # add one colum about label for excel 
    # add_text_excel(label_list, label2text_dict) 

    # generate single label for model training 
    data = multi_to_single(url_list, label_list, label2text_dict, wav_path)