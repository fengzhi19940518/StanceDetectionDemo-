import re
import jieba
def clean_str(string):
    string = re.sub(r"#", "", string)
    string = re.sub(r"/", "", string)
    string = re.sub(r"//", "", string)
    string = re.sub(r"…", "", string)
    string = re.sub(r"!", "",string)
    string = re.sub(r"【", "", string)
    string = re.sub(r"】", "", string)
    string = re.sub(r"。。。", "", string)
    return string.strip()

def read_chinese(path):
    file = open(path,'r',encoding="utf-8")
    text = []
    for line in file.readlines():
        line = clean_str(line)
        line = " ".join(jieba.cut(line.strip())).strip().split('\t')
        line = line[1:]
        text.append(line)
    # print(text)
    topics = [' IphoneSE ',' 春节 放鞭炮 ',' 俄罗斯 在 叙利亚 的 反恐 行动 ',' 开放 二胎 ',' 深圳 禁摩 限电 ',]
    topicList = []
    textList = []
    labelList =[]
    for line in text:
        # print(line)
        for i in range(0,2):
            topic1 = "".join(line[:i])
            if topic1 in topics:
                topicList.append(line[: i])
                textList.append(line[i:-1])
                labelList.append(line[-1].strip().split(" "))
            else:
                continue
    print(topicList)
    print(textList)
    print(len(labelList))
    # print(text)
# path1 ='./StanceDetectionDataset/dev.txt'
path2 = "./StanceDetectionDataset/NLPCC2016_Stance_Detection_Task_A_Traindata.txt"
# read_chinese(path2)

string = "123 ####一定aaa6s。。。要fx出去？/#， //。"
string = re.sub(r"\d","",string)
string = re.sub(r"#","",string)
string = re.sub(r'，',"",string)
string = re.sub(r'？',"",string)
string = re.sub(r"/","",string)
string = re.sub(r"//","",string)
string = re.sub(r"…","",string)
string = re.sub(r"。。。","",string)

