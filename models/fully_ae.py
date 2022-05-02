import torch
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 2                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_hidden_1 = 256                                          # 隐藏层大小
        self.input_features = 173056
        self.n_gram_vocab = 250499                                      # ngram 词表大小

class Model(torch.nn.Module):

        def __init__(self, config):
                super(Model, self).__init__()
                self.linear_1 = torch.nn.Linear(config.input_features, 256)
                ### DECODER
                self.linear_2 = torch.nn.Linear(256, config.input_features)


        def forward(self, x):
                encoded = self.linear_1(x)
                encoded = F.leaky_relu(encoded)
                ### DECODER
                logits = self.linear_2(encoded)
                decoded = torch.sigmoid(logits)

                return decoded