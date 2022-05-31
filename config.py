'''
Description: 
Version: 1.0
Autor: Chasey
Date: 2022-05-28 11:57:38
LastEditTime: 2022-05-31 01:42:42
'''
# 配置参数
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object): 
    def __init__(self,embeddingSize):
        self.embeddingSize = embeddingSize
        self.hiddenSizes = 256  # LSTM结构的神经元个数
        self.dropoutKeepProb = 0.2
        self.l2RegLambda = 0.1
        self.epsilon = 5

class Config(object):
    def __init__(self,seqLength,embeddingSize):
        self.sequenceLength = seqLength
        self.batchSize = 128
        self.rate = 0.8  # 训练集的比例
        self.training = TrainingConfig()
        self.model = ModelConfig(embeddingSize)
