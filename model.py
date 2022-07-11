from unicodedata import bidirectional
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import data2num
import word2vec



class BiLSTMmodel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size = 100, num_classes = 9, input_size = 50,) -> None:

        super().__init__()

        self.hidden_size = hidden_size

        vocab_size=embedding_matrix.shape[0]
        vector_size=embedding_matrix.shape[1]
 
        self.embedding=nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix),freeze=True)

        self.embedding.weight.requires_grad=False

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size ,bidirectional=True, batch_first=True)

        self.label = nn.Linear(2*hidden_size, num_classes)

        self.softmax = nn.Softmax(dim=2)


    def forward(self, x, l):

        # h_0 = Variable(torch.zeros(2, x.shape[0], self.hidden_size))

        # c_0 = Variable(torch.zeros(2, x.shape[0], self.hidden_size))

        x = self.embedding(x)

        x = x.to(torch.float32)

        x_packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=l, batch_first=True, enforce_sorted=False)

        output, (final_hidden_state, final_cell_state) = self.lstm(x_packed_input)

        output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = output[0]
        
        output = self.label(output)

        output_sm = self.softmax(output)
        
        return output_sm
    
    def printShape(self,x):
        print(x.shape)
        return x.shape




# embedding = word2vec.embedding(gt_dir = "data", filename = "glove.6B.50d.txt")

# test_model = BiLSTMmodel(embedding_matrix=embedding.embs_npa,hidden_size=100, num_classes=9, input_size=50)
# emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding.embs_npa),freeze=True)
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
# docs = ['on your mark', 
#         'get set',
#         'go']
# wo = []
# for seq in docs:
#     tmp = []
#     for word in seq.split(' '):
#         tmp.append(word)
#     wo.append(torch.IntTensor(data2num(tmp)))


# x =  pad_sequence(wo, batch_first=True, padding_value=0, )
# seq_len=torch.IntTensor(list(map(len,wo))) 

# out = test_model(x,seq_len)
# print(out.shape) 



