import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import random
from loadEmbedding import LoadEmbedding
torch.manual_seed(123)
random.seed(123)
class BiLSTM(nn.Module):
    def __init__(self,params):
        super(BiLSTM,self).__init__()
        self.params = params
        # self.embedding_text = nn.Embedding(params.text_num, params.embed_dim)
        # self.embedding_topic = nn.Embedding(params.topic_num, params.embed_dim)
        self.embedding_text = LoadEmbedding(params.text_num,params.embed_dim)
        self.embedding_text.load_pretrained_embedding(params.load_embedding_path,
                                                      params.text_dict,
                                                      params.save_text_embedding,
                                                      binary=True)
        self.embedding_topic = LoadEmbedding(params.topic_num,params.embed_dim)
        self.embedding_topic.load_pretrained_embedding(params.load_embedding_path,
                                                       params.topic_dict,
                                                       params.save_topic_embedding,
                                                       binary=True)

        # print(self.embedding_topic)
        self.hidden_size = params.hidden_size

        # if params.use_embedding is True:
        #     # pretrain_weight_text= np.array(params.pretrain_embed_text)
        #     # self.embedding_text.weight.data.copy_(torch.from_numpy(pretrain_weight_text))
        #     # pretrain_weight_topic = np.array(params.pretrain_embed_topic)
        #     self.embedding_topic.weight.data.copy_(params.pretrain_embed_topic)
        #     # self.embedding_text.weight.data.copy_(torch.FloatTensor(np.array(params.pretrain_embed_text)))
        #     self.embedding_text.weight.data.copy_(params.pretrain_embed_text)

        self.bilstm = nn.LSTM(params.embed_dim, self.hidden_size, dropout=params.dropout,num_layers=params.num_layers, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(self.hidden_size * 4, params.hidden_size // 2)
        self.linear2 = nn.Linear(self.hidden_size // 2, params.label_num)
        # self.hidden = self.init_hidden(params.num_layers, params.batch_size)
    #  torch.__version__ =0.2.1+a4fc05a
    # def init_hidden(self,num_layers,batch_size):
    #
    #     return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_size)),
    #             Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_size)))
    #
    def forward(self,topic,text):
        # print("aaaa")
        x = self.embedding_topic(topic)
        y = self.embedding_text(text)

        # print("x",x)
        bilstm_out_topic, _ = self.bilstm(x)
        bilstm_out_text, _  = self.bilstm(y)

        bilstm_out_topic = torch.transpose(bilstm_out_topic, 1, 2)
        bilstm_out_text= torch.transpose(bilstm_out_text, 1, 2)

        tanh_out_topic = F.tanh(bilstm_out_topic)   #[16, 200, 6]
        tanh_out_text = F.tanh(bilstm_out_text)     #[16, 200, 35]

        pool_out_topic = F.max_pool1d(tanh_out_topic, tanh_out_topic.size(2))
        pool_out_text = F.max_pool1d(tanh_out_text , tanh_out_text.size(2))

        # print("pool_topic",pool_out_topic.size())
        # print("pool_text",pool_out_text.size())
        topic_text = torch.cat([pool_out_topic,pool_out_text],1)
        # print("topic_text",topic_text.size())
        # squ_out = pool_out_text.squeeze(2)
        # squ_out = pool_out_topic.squeeze(2)
        squ_out = topic_text.squeeze(2)
        logit = self.linear1(squ_out)
        tanh_out2 = F.tanh(logit)
        # relu_out = F.relu(logit)
        # logit = self.linear2(relu_out)
        logit = self.linear2(tanh_out2)
        # print("logitsss",logit)

        return logit