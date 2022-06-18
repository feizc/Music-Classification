import pandas as pd
import os 
import matplotlib.pyplot  as plt 
from data_preprocess import data_preprocess, label2text_dict_generate, multi_to_single
from collections import Counter 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def read_excel(excel_path): 
    df = pd.read_excel(excel_path, engine='openpyxl')
    label_list = df['类别'].values.tolist() 
    prediction_list = df['预测类别'].values.tolist() 

    refine_prediction_list = [] 
    for p in prediction_list: 
        refine_prediction_list.append(p.split()[0])
    return label_list, refine_prediction_list



def plot_stack_row(white_list, text_label_list, prediction_label): 
    unique_predict_list = list(set(prediction_label))
    print(unique_predict_list)

    predict2label_dict = dict()
    for p in unique_predict_list: 
        predict2label_dict[p] = [] 
    
    for i in range(len(prediction_label)): 
        predict = prediction_label[i] 
        label_list = text_label_list[i].split(',') 
        for label in label_list: 
            if label in white_list: 
                predict2label_dict[predict].append(label) 
    
    
    for key in predict2label_dict.keys(): 
        predict2label_dict[key] = Counter(predict2label_dict[key]) 
    
    print('predict to label dict: ', predict2label_dict) 

    # x -> unique_predict_list, y -> white_list 
    x_axis = unique_predict_list 
    y_axis_list = [] 
    for text in white_list: 
        y_axis = []
        for predict in x_axis: 
            p_counter = predict2label_dict[predict] 
            y_axis.append(p_counter[text]) 
        y_axis_list.append(y_axis) 
        print(text, y_axis)
    
    width = 0.35 
    fig, ax = plt.subplots(figsize=(10,4)) 
    bottom = [0]*len(x_axis)
    for i in range(len(y_axis_list)): 
        ax.bar(x_axis, y_axis_list[i], bottom=bottom, label=white_list[i]) 
        new_bottom = []
        for j in range(len(x_axis)):
            new_bottom.append(bottom[j] + y_axis_list[i][j]) 
        bottom = new_bottom
    ax.legend() 
    plt.show()



if __name__ == '__main__':  
    excel_path = './data/meituan/music.xlsx' 
    text_label_list, prediction_label = read_excel(excel_path) 
    
    category_excel_path = './data/meituan/category.xlsx'
    output_path = './data/meituan'
    wav_path = './data/meituan/wav_music' 

    url_list, label_list = data_preprocess(excel_path, output_path)  
    
    label2text_dict = label2text_dict_generate(category_excel_path) 

    # generate single label for model training 
    white_list = multi_to_single(url_list, label_list, label2text_dict, wav_path)  

    plot_stack_row(white_list, text_label_list, prediction_label)



