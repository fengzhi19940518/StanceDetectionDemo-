class Hyperparameter:
    def __init__(self):

        self.train_path = './StanceDetectionDataset/NLPCC2016_Stance_Detection_Task_A_traindata.txt'
        self.test_path = './StanceDetectionDataset/NLPCC_2016_Stance_Detection_Task_A_gold.txt'

        # self.load_embedding_path = './word_embedding/giga_cn50.w2v'
        self.load_embedding_path ="word_embedding/news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
        self.save_text_embedding="word_embedding/embed_text_pickle.pkl"
        self.save_topic_embedding ="word_embedding/embed_topic_pickle.pkl"
        self.set_padding = False
        self.embed_dim = 64
        self.hidden_size = 100
        self.num_layers = 1
        self.dropout = 0.5
        self.batch_size = 16
        self.learnRate = 0.001

        self.epochs = 1000

        self.text_num = 0
        self.topic_num = 0
        self.label_num = 0

        self.use_embedding = True
        self.use_lstm = True
        self.pretrain_embed_text = []
        self.pretrain_embed_topic = []

        self.train_size = 0
        self.dev_size = 0
        self.test_size = 0

        self.train_print_acc = 10
        self.train_test = 10
        self.cuda_use = True

        self.text_dict = None
        self.topic_dict = None