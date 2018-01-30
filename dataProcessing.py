import re
import linecache
import numpy as np
import jieba
import torch
from torch.autograd import Variable
torch.manual_seed(123)

class Alphabet:
    def __init__(self):
        self.id2string = []
        self.string2id = {}
def clean_str(string):
    string = re.sub(r"#", "", string)
    string = re.sub(r"/", "", string)
    string = re.sub(r"//", "", string)
    string = re.sub(r"…", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"【", "", string)
    string = re.sub(r"】", "", string)
    string = re.sub(r"。。。", "", string)
    return string.strip()

def read_corpus_chinese(path):
    file = open(path,'r',encoding="utf-8")
    text = []
    for line in file.readlines():
        line = clean_str(line)
        line = " ".join(jieba.cut(line.strip())).strip().split('\t')
        line = line[1:]
        text.append(line)
    topics = [' IphoneSE ',' 春节 放鞭炮 ',' 俄罗斯 在 叙利亚 的 反恐 行动 ',' 开放 二胎 ',' 深圳 禁摩 限电 ',]
    topicList = []
    textList = []
    labelList =[]
    for line in text:
        for i in range(0,2):
            topic1 = "".join(line[:i])
            if topic1 in topics:
                topicList.append(line[: i])
                textList.append(line[i:-1])
                labelList.append(line[-1].strip().split(" "))
            else:
                continue
    text = []
    topic = []
    for i in textList:
        if len(i) != 0:
            s = i[0].strip().split(" ")
            temp = []
            for j in range(len(s)):
                if s[j] != "":
                    temp.append(s[j])
            text.append(temp)
    for i in topicList:
        s = i[0].strip().split(" ")
        temp = []
        for j in range(len(s)):
            if s[j] != "":
                temp.append(s[j])
        topic.append(temp)
    # print(topic)
    # print(text)
    # print(labelList)
    return topic, text, labelList

def createAlphabet(train_data):
    initAlpha = Alphabet()
    initAlpha.id2string.append('unk')
    initAlpha.id2string.append('<pad>')
    initAlpha.string2id['unk'] = 0
    initAlpha.string2id['<pad>'] = len(train_data) - 1
    for id in range(len(train_data)):
        for w in train_data[id]:
            if w not in initAlpha.id2string:
                initAlpha.id2string.append(w)
                initAlpha.string2id[w] = len(initAlpha.id2string) - 1
    # print("string2id", initAlpha.string2id)
    # print("id2string", initAlpha.id2string)
    print("complete to createAlphabet......")
    return  initAlpha.string2id,initAlpha.id2string
def createAlphabet1(train_data):
    initAlpha = Alphabet()
    initAlpha.id2string.append('unk')
    initAlpha.string2id['unk'] = 0
    for sen in train_data:
        for w in sen:
            if w not in initAlpha.string2id.keys():
                initAlpha.id2string.append(w)
                initAlpha.string2id[w] = len(initAlpha.id2string) - 1
    # print("string2id", initAlpha.string2id)
    # print("id2string", initAlpha.id2string)
    # initAlpha.id2string.append('<pad>')
    # initAlpha.string2id['<pad>'] = len(initAlpha.string2id)
    print("complete to createAlphabet......")
    return initAlpha.string2id, initAlpha.id2string

def createAlphabet_label_single(train_data):
    print("Greate label Alphabet....")
    initAlpha = Alphabet()
    for index in range(len(train_data)):
        for w in train_data[index]:
            if w not in initAlpha.id2string:
                initAlpha.id2string.append(w)
                initAlpha.string2id[w] = len(initAlpha.id2string) - 1
    print("string2id",initAlpha.string2id)
    print("id2string",initAlpha.id2string)
    print('Complete to create alphabet label....')
    return initAlpha.string2id,initAlpha.id2string


def create_one_batch(data, string2id, id2string):
    data_index = []
    for index in range(len(data)):
        sen_index = []
        for w in data[index]:
            if w not in string2id.keys():
                sen_index.append(string2id['unk'])
            else:
                sen_index.append(string2id[w])
        data_index.append(sen_index)
    data_index, string2id, id2string, max_len = pad(data_index, string2id, id2string)

    return data_index, string2id, id2string

def create_one_label_batch(data, string2id, id2string):
    data_index = []
    for index in range(len(data)):
        sen_index = []
        for w in data[index]:
            # print("string2id[w]",string2id[w])
            sen_index.append(string2id[w])

        data_index.append(sen_index)
    data_index, string2id, id2string, max_len = pad(data_index, string2id, id2string)
    return data_index, string2id, id2string

def pad(data_index, string2id, id2string, pad_token = '<pad>'):
    max_len = seek_max_len(data_index)
    for i in range(len(data_index)):
        sen_index = data_index[i]
        if len(sen_index) < max_len:
            if pad_token not in id2string:
                id2string.append(pad_token)
                string2id[pad_token] = len(id2string) -1
            for j in range(max_len-len(sen_index)):
                sen_index.append(string2id[pad_token])

    return data_index,string2id,id2string,max_len

def seek_max_len(data_index):
    max_len = 0
    for i in range(len(data_index)):
        if(len(data_index[i])> max_len):
            max_len = len(data_index[i])
    return max_len

def load_embedding(word2id,params):
    print("Loading embedding...")
    path = params.load_embedding_path
    t = params.embed_dim
    words = []
    words_dict ={}
    with open(path,'r',encoding='utf-8') as file:
        for line in file.readlines():
            text = line.strip().split(' ')
            word = text[0]
            nums = text[1:]
            nums = [float(e) for e in nums]
            words.append(word)
            words_dict[word] = nums
        count_list = []
        count = 0
        dict_cat = []
        for word in word2id:
            if word in words_dict.keys():
                count += 1
                dict_cat.append(words_dict[word])
            else:
                dict_cat.append([0.0] * t)
                count += 1
                count_list.append(count - 1)
        count_data = len(word2id) - len(count_list)
        sum = []
        for j in range(t):
            sum_col = 0.0
            for i in range(len(dict_cat)):
                sum_col += dict_cat[i][j]
                sum_col = float(sum_col / count_data)
                sum_col = round(sum_col, 6)
            sum.append(sum_col)
        for i in range(len(count_list)):
            dict_cat[count_list[i]] = sum
    print("Complete to load embedding.\n")
    return dict_cat

def load_pretrained_emb_avg(path, text_field_words_dict, set_padding = False):
    padID = text_field_words_dict['<pad>']
    embedding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) == 1:
                embedding_dim = line_split[0]
                break
            elif len(line_split) == 2:
                embedding_dim = line_split[1]
                break
            else:
                embedding_dim = len(line_split) - 1
                break
    word_count = len(text_field_words_dict)
    print('The number of words is ' + str(word_count))
    print('The dim of pretrained embedding is ' + str(embedding_dim) + '\n')
    embeddings = np.zeros((word_count, embedding_dim))
    inword_list = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split(' ')
            index = text_field_words_dict.get(values[0])   # digit or None
            if index:
                vector = np.array(values[1: -1], dtype='float32')
                embeddings[index] = vector
                inword_list.append(index)
    sum_col = np.sum(embeddings, axis=0) / len(inword_list)   # 按列求和，再求平均
    for i in range(len(text_field_words_dict)):
        if i not in inword_list and i != padID:
            embeddings[i] = sum_col
    return torch.from_numpy(embeddings).float()

def create_batches(train_topic_var, train_text_var, train_label_var, batch_size):
    print("Creating batches....")
    train_topic_iter = []
    train_text_iter = []
    train_label_iter = []
    if train_text_var.size(0) % batch_size == 0:
        batch_num = train_text_var.size(0) // batch_size
    else:
        batch_num = train_text_var.size(0) // batch_size + 1

    for i in range(batch_num):
        line_iter = []
        for j in range(batch_size):
            if (i * batch_size + j +1) <= len(train_topic_var):
                line_iter.append(train_topic_var.data[i * batch_size + j].tolist())
        train_topic_iter.append(Variable(torch.LongTensor(line_iter)))

    for i in range(batch_num):
        line_iter = []
        for j in range(batch_size):
            if (i * batch_size + j + 1) <= len(train_text_var):
                line_iter.append(train_text_var.data[i * batch_size + j].tolist())
        train_text_iter.append(Variable(torch.LongTensor(line_iter)))

    for i in range(batch_num):
        line_iter = []
        for j in range(batch_size):
            if (i * batch_size + j +1) <= len(train_label_var):
                line_iter.append(train_label_var.data[i * batch_size + j].tolist())
        train_label_iter.append(Variable(torch.LongTensor(line_iter)))
    # print("train_topic_iter",train_topic_iter)
    # print("train_text_iter", train_text_iter)
    # print("train_label_iter",train_label_iter)
    print("Complete to create batches..")
    return train_topic_iter, train_text_iter, train_label_iter

def cal_Eval_Score(p, r, c):
    if p != 0:
        precision = c / p
    else:
        precision = 0
    if r != 0:
        recall = c / r
    else:
        recall = 0
    if (precision + recall) != 0:
        fscore = 2 * c  / (p + r)
    else:
        fscore = 0
    return precision, recall, fscore
