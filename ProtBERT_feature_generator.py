from time import time
import torch
from transformers import BertModel, BertTokenizer
import re
import os
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle
import pandas as pd

# generate_protbert_features 函数用于从一组蛋白质序列生成特征，这些蛋白质序列从一个 CSV 文件中读取。
# 该函数使用 BertTokenizer 来对序列进行分词，然后使用 BERT 模型（在这里是 ProtBERT，一种专门针对蛋白质序列的 BERT 变体）来生成序列的嵌入表示。
# 最后，这些嵌入表示被保存在一个 pickle 文件中
def generate_protbert_features(file):
    path = 'D:/fengzhen/EnsemPPIS-master/feature_generator/'    # 设置文件路径
    t0=time()      # 开始计时

    modelFilePath = path+'pytorch_model.bin'    # 模型、配置和词汇文件路径
    configFilePath = path+'config.json'
    vocabFilePath = path+'vocab.txt'
        
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )    # 初始化分词器
    model = BertModel.from_pretrained(path)    # 从预训练模型加载 ProtBERT 模型
    # device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')    # 设置设备为 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为 GPU 或 CPU

    model = model.to(device)
    model = model.eval()

    sequences = []
    df = pd.read_csv(file, sep=',')    # 读取蛋白质序列数据
    sequences =df['sequence'].values.tolist()
        
    sequences_Example = [' '.join(list(seq)) for seq in sequences]    # 对每个蛋白质序列进行预处理，包括将字符分开并替换非标准氨基酸
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]

    all_protein_features = []

    for i, seq in enumerate(sequences_Example):    # 对每个序列生成嵌入表示
        print(i)
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=False)    # 编码序列
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():    # 生成序列嵌入
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):     # 提取有效嵌入部分
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
            
            print(features.__len__())
        # print(all_protein_sequences['all_protein_complex_pdb_ids'][i])
    #     print(features)
        all_protein_features += features

    pickle.dump(all_protein_features, open('../data_cache/74_encode_data.pkl', 'wb'))     # 保存所有蛋白质的特征
    ##["dset186","dset164","dset72"]_encode_data.pkl

    print('Total time spent for ProtBERT:',time()-t0)

if __name__ == "__main__":

    file =  'D:/fengzhen/EnsemPPIS-master/data_cache/422_name_seq_label_74.csv'

    generate_protbert_features(file)
