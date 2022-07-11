import numpy as np
import pandas as pd
import torch
from dataset import ConllDataset
from eval import F1Score
from model import BiLSTMmodel
import word2vec

EPOCH_NUM = 20
Device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(pred, gold, length):
    correct = 0
    for i in range(pred.shape[0]):
        pred_b = pred[i]
        pred_b = torch.max(pred_b, 0)[1]
        correct_b = (pred_b[:length[i]] == gold[i,:length[i]]).sum().item()
        correct += correct_b/length[i]
    return correct / pred.shape[0]




def train():
    print("Training started")
    embedding = word2vec.embedding(gt_dir = "data", filename = "glove.6B.50d.txt")
    model = BiLSTMmodel(embedding_matrix=embedding.embs_npa,hidden_size=100, num_classes=9, input_size=50)
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = ConllDataset('train', './data/train.conll')
    dev_dataset = ConllDataset('dev', './data/dev.conll')
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean', weight=train_dataset.class_weights)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=1, shuffle=True)
    f1_score_macro = F1Score(average='macro')
    train_loss = []
    # train_acc = []
    dev_f1 = []
    dev_acc = []

    for epoch in range(EPOCH_NUM):

        print("Epoch: ", epoch+1)
        for step, (x, y, l) in enumerate(train_loader):
            
            x = x.to(Device)
            y = y.to(Device)
            l = l.to(Device)
            out = model(x,l)
            out = out.permute(0,2,1)
            loss = 0
            for i in range(out.shape[0]):
                loss += loss_func(torch.unsqueeze(out[i,:,:l[i]],dim=0), torch.unsqueeze(y[i,:l[i]],dim=0))
                # loss_batch.append(loss)
            loss = loss/out.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                print("Epoch: ", epoch+1, "| Step: ", step, "| Loss: ", loss.item())
        train_loss.append(loss.item())
                # train_acc.append(accuracy(out, y))

        with torch.no_grad():
            model.eval()
            dev_acc_batch = []
            dev_f1_batch = []
            print('Dev ----------------------------------------------------------')
            for step, (x, y, l) in enumerate(dev_loader):
                x = x.to(Device)
                y = y.to(Device)
                l = l.to(Device)
                out = model(x,l)
                out = out.permute(0,2,1)
                acc = accuracy(out, y, l)
                dev_acc_batch.append(acc)
                pred_b = out[0]
                pred_b = torch.max(pred_b, 0)[1]
                dev_f1_batch.append(f1_score_macro(pred_b, y[0,:l[0]], l))
            print("Epoch: ", epoch+1, "| Step: ", step, "| acc: ", np.mean(dev_acc_batch), "| f1: ", np.mean(dev_f1_batch))
            print('----------------------------------------------------------------')
            dev_acc.append(np.mean(dev_acc_batch))
            dev_f1.append(np.mean(dev_f1_batch))
            torch.save(model.state_dict(), './model/model_epoch_' + str(epoch+1) + '.pt')
        
        model.train()
    
    dev_f1 = pd.DataFrame(np.array(dev_f1))
    dev_acc = pd.DataFrame(np.array(dev_acc))
    train_loss = pd.DataFrame(np.array(train_loss))
    # train_acc = pd.DataFrame(np.array(train_acc))

    dev_f1.to_csv('dev_f1.csv', index=False)
    dev_acc.to_csv('dev_acc.csv', index=False)
    train_loss.to_csv('train_loss.csv', index=False)
    # train_acc.to_csv('train_acc.csv', index=False)
    print("Data saved")
    

        
    


if __name__ == '__main__':
    train()