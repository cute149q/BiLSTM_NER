

import numpy as np


class embedding():
    
    def __init__(self, gt_dir, filename):

        self.file_path = f"{gt_dir}/{filename}"
        self.vocab, self.embd = self.readFile(self.file_path)
        self.vocab_npa = np.array(self.vocab)
        self.embs_npa = np.array(self.embd, dtype=np.float32)
        #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
        self.vocab_npa = np.insert(self.vocab_npa, 0, '<pad>')
        self.vocab_npa = np.insert(self.vocab_npa, 1, '<unk>')
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab_npa)}    #Dict word to index
        # print(self.vocab_npa[:10])

        pad_emb_npa = np.zeros((1,self.embs_npa.shape[1]))   #embedding for '<pad>' token.
        unk_emb_npa = np.mean(self.embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
        #insert embeddings for pad and unk tokens at top of embs_npa.
        self.embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,self.embs_npa))


    def readFile(self, filename):
    # Dict word to 100d vector
        embed = []
        vocab = []
        with open(filename, 'r') as f:
            for line in f:
                s = line.strip().split(' ')
                vocab.append(s[0])
                embed.append(s[1:])
            return vocab,embed

