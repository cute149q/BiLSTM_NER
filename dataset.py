import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import word2vec
# from sklearn.utils import class_weight


label_to_idx = {
    'O': 0,
    'B-ORG': 1,
    'I-ORG': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8,
    '<pad>': 9,
}

Glove = word2vec.embedding('data', 'glove.6B.50d.txt')

def data2num(data):
    data_num = []
    for i in range(len(data)):
        if data[i] in Glove.vocab_dict:
            data_num.append(Glove.vocab_dict[data[i]])
        else:
            data_num.append(Glove.vocab_dict['<unk>'])
    return data_num

def label2num(label):
    label_num = []
    for i in range(len(label)):
        label_num.append(label_to_idx[label[i]])
    return label_num

class ConllDataset(Dataset):

    def __init__(self, split, gt_dir):
        assert split in ['train', 'test', 'dev'], 'Only train, test and dev splits are implemented.'
        assert os.path.exists(gt_dir), 'gt_dir path does not exist: {}'.format(gt_dir)
        self.split = split
        self.gt_dir = gt_dir
        list_use = [0,3]
        self.data_path = f'./data/{split}.conll'
        self.data = pd.read_csv(self.data_path,on_bad_lines='skip', skip_blank_lines=False, sep='\t', names=["word", "dummy1", "dummy2", "NE"], header=0, usecols=list_use)
        self.word = []
        self.label = []
        self.label_weight = []
        self.word_len = []
        word_temp = []
        lable_temp = []

        for index, row in self.data.iterrows():
            # print(row['word'], row['NE'])
            
            if(pd.isna(row['word']) and pd.isna(row['NE']) and len(word_temp)!=0 and len(lable_temp)!=0):
                # print(word_temp)
                self.word.append(torch.LongTensor(data2num(word_temp)))
                self.label.append(torch.LongTensor(label2num(lable_temp)))
                word_temp = []
                lable_temp = []
            elif(row['word']!='\t' and not pd.isna(row['word']) and not pd.isna(row['NE'])):
                    word_temp.append(row['word'])
                    lable_temp.append(row['NE'])
                    self.label_weight.append(label_to_idx[row['NE']])

        

        self.len = len(self.word)

        self.word_padded = torch.nn.utils.rnn.pad_sequence(self.word, batch_first=True)
        self.label_padded = torch.nn.utils.rnn.pad_sequence(self.label, batch_first=True, padding_value=9)
        self.word_len = [len(x) for x in self.word]
        if split == 'train':
            # class_weights=class_weight.compute_class_weight(class_weight='balanced',classes = np.array([0,1,2,3,4,5,6,7,8]),y = np.array(self.label_weight))
            # class_weights=torch.tensor(class_weights,dtype=torch.float)
            # self.class_weights=class_weights
            class_sample_count = np.unique(self.label_weight, return_counts=True)[1]
            weight = 1. / class_sample_count
            self.class_weights = torch.FloatTensor(weight)

    def __getitem__(self, index):

        return self.word_padded[index], self.label_padded[index], self.word_len[index]

    def __len__(self):
        return self.len
    


if __name__ == "__main__":
    print("test")
