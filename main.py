import dataProcessing
import hyperparameter
import torch
from torch.autograd import Variable
import model_BiLSTM
import train2
import random
torch.manual_seed(123)

if __name__=='__main__':
    params = hyperparameter.Hyperparameter()
    # topicList, textList, labelList, topic_text = dataProcessing.read_corpus(params.train_path)
    topicList, textList, labelList = dataProcessing.read_corpus_chinese(params.train_path)
    # print(textList)
    # random.shuffle(Init_topicList)
    # random.shuffle(Init_textList)
    # random.shuffle(Init_labelList)
    # dev_supply_topic = Init_topicList[:199]
    # dev_supply_text = Init_textList[:199]
    # dev_supply_label = Init_labelList[:199]
    #
    # test_topic = Init_topicList[200:800]
    # test_text = Init_textList[200:800]
    # test_label = Init_labelList[200:800]
    #
    # topicList = Init_topicList[801:]
    # textList = Init_textList[801:]
    # labelList = Init_labelList[801:]

    # dev_topic, dev_text, dev_label, dev_topic_text = dataProcessing.read_corpus_chinese(params.dev_path)
    # dev_topic = dev_topic + dev_supply_topic
    # dev_text = dev_text + dev_supply_text
    # dev_label = dev_label + dev_supply_label
    # # print("dev_label",dev_label)
    test_topic, test_text, test_label = dataProcessing.read_corpus_chinese(params.test_path)
    params.train_size = len(topicList)
    print("len_train",len(topicList))
    params.test_size = len(test_topic)
    print("len_test",len(test_topic))

    topic_word2id, topic_id2word = dataProcessing.createAlphabet1(topicList)
    params.topic_dict = topic_word2id
    text_word2id, text_id2word = dataProcessing.createAlphabet1(textList)
    params.text_dict = text_word2id
    label2id, id2label = dataProcessing.createAlphabet_label_single(labelList)

    topic_index, topic_word2id, topic_id2word = dataProcessing.create_one_batch(topicList, topic_word2id, topic_id2word)
    text_index, text_word2id, text_id2word = dataProcessing.create_one_batch(textList, text_word2id, text_id2word)
    label_index, _, _ = dataProcessing.create_one_label_batch(labelList, label2id, id2label)

    # dev_topic_index, topic_word2id, topic_id2word = dataProcessing.create_one_batch(dev_topic, topic_word2id, topic_id2word)
    # dev_text_index, dev_text_word2id, text_id2word = dataProcessing.create_one_batch(dev_text,text_word2id, text_id2word)
    # dev_label_index, dev_label2id, dev_id2label = dataProcessing.create_one_label_batch(dev_label, label2id, id2label)

    test_topic_index, topic_word2id, topic_id2word = dataProcessing.create_one_batch(test_topic, topic_word2id, topic_id2word)
    test_text_index, text_word2id, text_id2word = dataProcessing.create_one_batch(test_text, text_word2id, text_id2word)
    test_label_index, test_label2id, test_id2label = dataProcessing.create_one_label_batch(test_label, label2id, id2label)

    params.topic_num = len(topic_id2word)
    params.text_num = len(text_id2word)
    params.label_num = len(id2label)
    # print(params.label_num)
    # if params.use_embedding is True:
        # params.pretrain_embed_text = dataProcessing.load_embedding(word2id, params)
        # params.pretrain_embed_topic = dataProcessing.load_embedding(text_word2id,params)
        # params.pretrain_embed_text = dataProcessing.load_pretrained_emb_avg(params.load_embedding_path, text_word2id)
        # params.pretrain_embed_topic = dataProcessing.load_pretrained_emb_avg(params.load_embedding_path, topic_word2id)
    # print("sdss",len(train_index))
    train_topic_var = Variable(torch.LongTensor(topic_index))
    train_text_var = Variable(torch.LongTensor(text_index))
    train_label_var = Variable(torch.LongTensor(label_index))

    # dev_topic_var = Variable(torch.LongTensor(dev_topic_index))
    # dev_text_var = Variable(torch.LongTensor(dev_text_index))
    # dev_label_var = Variable(torch.LongTensor(dev_label_index))

    test_topic_var = Variable(torch.LongTensor(test_topic_index))
    test_text_var = Variable(torch.LongTensor(test_text_index))
    test_label_var = Variable(torch.LongTensor(test_label_index))
    # # print("ssss",test_topic_var)
    # print(test_text_var)
    # print(test_label_var)

    # dev_iter = dataProcessing.create_batches(dev_topic_var, dev_text_var, dev_label_var, params.batch_size)
    # print(dev_iter)
    test_iter = dataProcessing.create_batches(test_topic_var, test_text_var, test_label_var, params.batch_size)
    # print(test_iter)
    # print("train_var",train_topic_var)          #2414x6
    # print("train_text_var",train_text_var)      #2414x35
    # print("train_label_var",train_label_var)    #2414x1
    if params.use_lstm is True:
        model = model_BiLSTM.BiLSTM(params)
        if params.cuda_use is True:
            model = model.cuda()
        train2.train(train_topic_var, train_text_var, train_label_var, model, label2id, id2label, params,test_iter)
