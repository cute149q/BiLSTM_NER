import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dev_f1 = pd.read_csv('dev_f1.csv')
dev_acc = pd.read_csv('dev_acc.csv')
train_loss = pd.read_csv('train_loss.csv')

plt.plot(dev_f1.index+1, dev_f1['0'], label='dev_f1')
plt.plot(dev_acc.index+1, dev_acc['0'], label='dev_acc')
plt.plot(train_loss.index+1, train_loss['0'], label='train_loss')
plt.xticks(range(1,20,2))
plt.xlabel('Epoch')
plt.legend()

plt.show()


