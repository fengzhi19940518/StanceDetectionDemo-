import torch
import random
import torch.nn.functional as F
torch.manual_seed(123)
random.seed(123)
import dataProcessing
import numpy as np
import sys

def train (train_topic_var, train_text_var, train_label_var, model, label2id, id2label, params,test_iter):
    optimizer = torch.optim.Adam(model.parameters(),lr = params.learnRate)
    steps = 0
    max_test_value = 0.0
    print(model)
    for epoch in range(1, params.epochs+1):
        train_len = train_text_var.size(0)
        perm_list = torch.randperm(train_len)
        train_topic_var = train_topic_var[perm_list]
        train_text_var = train_text_var[perm_list]
        train_label_var = train_label_var[perm_list]
        train_topic_iter,train_text_iter,train_label_iter = dataProcessing.create_batches(train_topic_var, train_text_var, train_label_var, params.batch_size)

        for index in range(len(train_label_iter)):
            # print(len(train_label_iter))    #151
            model.zero_grad()
            # if train_label_iter[index].size(0) != params.batch_size:
            #     model.hidden = model.init_hidden(params.num_layers, train_label_iter[index].size(0))
            # else:
            #     model.hidden = model.init_hidden(params.num_layers, params.batch_size)
            # print("train_topic_iter",len(train_topic_iter))     #151
            # print("train_text_iter",train_text_iter[index])     #train_text_iter[index] =35*16
            if params.cuda_use is True:
                train_topic_iter[index] = train_topic_iter[index].cuda()
                train_text_iter[index] = train_text_iter[index].cuda()
                train_label_iter[index] = train_label_iter[index].cuda()
            logit = model(train_topic_iter[index], train_text_iter[index])
            # print("logit",logit)  #16*3
            # print("train_label_iter",train_label_iter[index])   #16*1
            batch_size = train_label_iter[index].size(0)
            # print("logit",logit) dex]",train_label_iter[index])
            loss = F.cross_entropy(logit, train_label_iter[index].squeeze(1))
            # print("ssss")
            loss.backward()
            optimizer.step()
            steps += 1
            evalValue = []
            for i in range(3):
                evalParams = []
                for j in range(3):
                    evalParams.append(0)
                evalValue.append(evalParams)
            correct_num = 0
            if steps % params.train_print_acc == 0:
                predict = torch.max(logit, 1)[1].data.tolist()
                # print(predict)
                label_iter_index_list = train_label_iter[index].squeeze(1).data.tolist()
                # print("label",label_iter_index_list)
                for id in range(batch_size):
                    evalValue[0][predict[id]] += 1
                    evalValue[1][label_iter_index_list[id]] += 1
                    if predict[id] == label_iter_index_list[id]:
                        correct_num += 1
                        evalValue[2][predict[id]] +=1
                f_score_micro = correct_num / batch_size

                precision = []
                recall = []
                fscore = []
                for idx in range(3):
                    p, r, f = dataProcessing.cal_Eval_Score(evalValue[0][idx], evalValue[1][idx], evalValue[2][idx])
                    precision.append(p)
                    # print("precision",precision)
                    recall.append(r)
                    fscore.append(f)
                p_macro = np.mean(precision)
                r_macro = np.mean(recall)
                f_score_macro = np.mean(fscore)

                # print("\rBatch[{}] - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{}) ".format(
                #     steps, loss.data[0], (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num, batch_size))
                sys.stdout.write("\rBatch[{}] - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{}) \n".format(
                        steps, loss.data[0], (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num, batch_size))
                # print("\n")
            if steps % params.train_test == 0:
                # eval_dev(dev_iter, model, params, id2label, label2id)
                eval_test(test_iter, model, params, id2label, label2id)


def eval_dev(dev_iter, model, params, id2label, label2id):
    dev_topic_iter, dev_text_iter, dev_label_iter = dev_iter
    avg_loss = 0.0
    correct_num = 0
    evalValue = []
    for i in range(3):
        evalParams = []
        for j in range(3):
            evalParams.append(0)
        evalValue.append(evalParams)

    for index in range(len(dev_label_iter)):
        if params.cuda_use is True:
            dev_topic_iter[index] = dev_topic_iter[index].cuda()
            dev_text_iter[index] = dev_text_iter[index].cuda()
            dev_label_iter[index] = dev_label_iter[index].cuda()
        logit = model(dev_topic_iter[index],dev_text_iter[index])
        loss = F.cross_entropy(logit, dev_label_iter[index].squeeze(1))
        avg_loss += loss.data[0]
        batch_size = dev_label_iter[index].size(0)

        predict = torch.max(logit, 1)[1].data.tolist()
        label_iter_index_list = dev_label_iter[index].squeeze(1).data.tolist()
        # print("label",label_iter_index_list)
        for id in range(batch_size):
            evalValue[0][predict[id]] += 1
            evalValue[1][label_iter_index_list[id]] += 1
            if predict[id] == label_iter_index_list[id]:
                correct_num += 1
                evalValue[2][predict[id]] += 1

        precision = []
        recall = []
        fscore = []
        for idx in range(3):
            p, r, f = dataProcessing.cal_Eval_Score(evalValue[0][idx], evalValue[1][idx], evalValue[2][idx])
            precision.append(p)
            recall.append(r)
            fscore.append(f)
        p_macro = np.mean(precision)
        r_macro = np.mean(recall)
        f_score_macro = np.mean(fscore)

        f_score_micro = correct_num / params.dev_size
        avg_loss = avg_loss / params.dev_size

        # print("\rDev Evalution - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{}) ".format(
        #         avg_loss, (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num, params.dev_size))
        sys.stdout.write("\rDev Evalution - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{})".format(
                avg_loss, (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num, params.dev_size))
    print("\n")


def eval_test(test_iter, model, params, id2label, label2id):
    test_topic_iter, test_text_iter, test_label_iter = test_iter
    avg_loss = 0.0
    correct_num = 0
    evalValue = []
    for i in range(3):
        evalParams = []
        for j in range(3):
            evalParams.append(0)
        evalValue.append(evalParams)

    for index in range(len(test_label_iter)):
        if params.cuda_use is True:
            test_topic_iter[index] = test_topic_iter[index].cuda()
            test_text_iter[index] = test_text_iter[index].cuda()
            test_label_iter[index] = test_label_iter[index].cuda()
        logit = model(test_topic_iter[index],test_text_iter[index])
        loss = F.cross_entropy(logit, test_label_iter[index].squeeze(1))
        avg_loss += loss.data[0]
        batch_size = test_label_iter[index].size(0)

        predict = torch.max(logit, 1)[1].data.tolist()
        label_iter_index_list = test_label_iter[index].squeeze(1).data.tolist()
        # print("label",label_iter_index_list)
        for id in range(batch_size):
            evalValue[0][predict[id]] += 1
            evalValue[1][label_iter_index_list[id]] += 1
            if predict[id] == label_iter_index_list[id]:
                correct_num += 1
                evalValue[2][predict[id]] += 1

        precision = []
        recall = []
        fscore = []
        # for idx in range(3):
        #     p, r, f = dataProcessing.cal_Eval_Score(evalValue[0][idx], evalValue[1][idx], evalValue[2][idx])
        #     precision.append(p)
        #     recall.append(r)
        #     fscore.append(f)
        # p_macro = np.mean(precision)
        # r_macro = np.mean(recall)
        # f_score_macro = np.mean(fscore)
        for idx in range(3):
            p, r, f = dataProcessing.cal_Eval_Score(evalValue[0][idx], evalValue[1][idx], evalValue[2][idx])
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            # print("fscore",fscore)
        p_macro = np.mean(precision)
        r_macro = np.mean(recall)
        f_score_macro = np.mean(fscore)
        # f_score_macro = (fscore[1] + fscore[2]) / 2

        f_score_micro = correct_num / params.test_size
        avg_loss = avg_loss / params.test_size

        # print("\rTest Evalution - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{}) ".format(
        #         avg_loss, (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num, params.test_size))
        sys.stdout.write("\rTest Evalution - loss: {:.6f} f_score_micro: {:.4f}% p_macro: {:.4f} r_macro: {:.4f} f_score_macro: {:.4f}% ({}/{})".format(
                avg_loss, (f_score_micro * 100), p_macro, r_macro, (f_score_macro * 100), correct_num,
                params.test_size))
    print("\n")








