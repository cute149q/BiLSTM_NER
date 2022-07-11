import numpy as np
import pandas as pd
import torch
from dataset import ConllDataset
from eval import F1Score
from model import BiLSTMmodel
import word2vec



if __name__ == '__main__':
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding = word2vec.embedding(gt_dir = "data", filename = "glove.6B.50d.txt")
    model = BiLSTMmodel(embedding_matrix=embedding.embs_npa,hidden_size=100, num_classes=9, input_size=50)
    model = model.to(Device)
    model.load_state_dict(torch.load("./model/model_epoch_20.pt"))
    model.eval()

    test_dataset = ConllDataset('test', './data/test.conll')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    f1_score_macro = F1Score(average='macro')
    test_f1 = []

    print("Testing started with model_epoch_20.pt with test size: ", len(test_dataset))
    for step, (x, y, l) in enumerate(test_loader):
        x = x.to(Device)
        y = y.to(Device)
        l = l.to(Device)
        out = model(x,l)
        out = out.permute(0,2,1)
        pred_b = out[0]
        pred_b = torch.max(pred_b, 0)[1]
        test_f1.append(f1_score_macro(pred_b, y[0,:l[0]], l))
    print("Test F1: ", np.mean(test_f1))
