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



if __name__ == '__main__': 
    excel_path = './data/meituan/music.xlsx' 
    output_path = './data/meituan'
    data_preprocess(excel_path, output_path)  
