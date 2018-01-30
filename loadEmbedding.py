import numpy as np
from numpy import *
import torch.nn as nn
import torch
import pickle
import os

class LoadEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(LoadEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.embedding_dict = {}

    def load_pretrained_embedding(self, file, model_dict, embed_pickle="embed_file.pkl",binary=False, encoding='utf8', datatype=float32):
        """
        :param file: pretrained embedding file path
        :param model_dict: features dict
        :param embed_pickle: save embed file
        :param binary: if the file is binary ,set binary True,else set False
        :param encoding: the default encoding is 'utf8'
        :param datatype: vector datatype , the default is float32
        :return:
        """
        if os.path.exists(embed_pickle):
            narray = pickle.load(open(embed_pickle,'rb'))
            self.weight = nn.Parameter(torch.FloatTensor(narray))
        else:
            with open(file, 'rb') as fin:
                header = str(fin.readline(), encoding).split()
                vocab_size = dim_size = 0
                if binary:
                    if header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        print("don't support this type")
                        exit(0)
                    binary_len = dtype(datatype).itemsize * int(dim_size)
                    for i in range(vocab_size):
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == b' ':
                                break
                            if ch == b'':
                                raise EOFError
                            if ch != b'\n':
                                word.append(ch)
                        word = str(b''.join(word), encoding)
                        weight = fromstring(fin.read(binary_len), dtype=datatype)
                        if word in model_dict:
                            self.embedding_dict[word] = weight
                else:
                    if header.__len__() == 1:
                        dim_size = int(header[0])
                        vocab_size = fin.readlines().__len__() + 1
                        fin.seek(0)
                    elif header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        vocab_size = fin.readline().__len__() + 1
                        dim_size = header[1:].__len__()
                        fin.seek(0)
                    for i in range(vocab_size):
                        data = str(fin.readline(), encoding).strip().split(' ')
                        word, weight = data[0], fromstring(' '.join(data[1:]), dtype=datatype, sep=' ')
                        if word in model_dict:
                            self.embedding_dict[word] = weight
            narray = np.empty((0, 0))
            num = 0
            for k, v in model_dict.items():
                if k in self.embedding_dict.keys():
                    temp = np.array([self.embedding_dict[k]])
                else:
                    temp = np.array([[random.uniform(-0.01, 0.01) for i in range(dim_size)]])
                if num == 0:
                    narray = temp
                    num += 1
                    continue
                narray = np.concatenate(([narray, temp]))
                num += 1
                # print("concatenate %d ,word : %s " % (num, k))
            pickle.dump(narray,open(embed_pickle,'wb'))
            self.weight = nn.Parameter(torch.FloatTensor(narray))

