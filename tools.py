'''
Description: 
Version: 1.0
Autor: Chasey
Date: 2022-05-28 11:56:49
LastEditTime: 2022-05-29 00:37:47
'''
import pandas as pd
import numpy as np
import re
import jieba
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# 正则匹配所有的非中文词 
sub_re='[^\u4e00-\u9fa5]'

class seqLength(object):
    def __init__(self, path, stopwordPath):
        self.path=path
        self.words_num=[]
        self.stopwordPath=stopwordPath
    
    def stopwordslist(self):
        stopwords = [line.strip() for line in open(self.stopwordPath, 'r',encoding="utf-8").readlines()]
        return stopwords

    # 对句子进行分词
    def seg_sentence(self, sentence):
        sentence_seged = jieba.lcut(sentence.strip())
        stopwords = self.stopwordslist() # 这里加载停用词的路径
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr
        
    def seq_len(self):
        """
        从csv文件中读取数据集
        """  
        df = pd.read_csv(self.path)
        review = df["reviewtext"].tolist()
        for line in review:#每次读一行
            line=re.sub(sub_re,'',line)
            #结巴分词
            cut_text=jieba.lcut(line)
            cut_list=[i for i in cut_text]
            num = len(cut_list)
            self.words_num.append(num)

        # 平均的长度
        mean_length=np.mean(self.words_num)
        # 最长的评价的长度
        max_length=np.max(self.words_num)
        print("the max length",max_length)
        self.sequenceLength=mean_length+2*np.std(self.words_num)
        self.sequenceLength=int(self.sequenceLength)
        return self.sequenceLength

    def plot_seqence(self):
        ##绘制sequence分布图
        plt.hist(self.words_num, bins = 100)
        plt.xlim((0,100))
        plt.ylabel('词频')
        plt.xlabel('序列长度')
        plt.title('评论数据集')
        #保存高清照片
        #plt.savefig('评论数据集序列长度分布图.png',dpi=500)
        plt.show()
        sum=0
        for num in self.words_num:
            if self.sequenceLength>num:
                flag=1
                sum+=flag
            else:
                flag=0
                sum+=flag
        #这个长度占所有长度得比例
        print("the proportion of length",np.round(sum/len(self.words_num),3))


def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    # 打乱顺序
    np.random.shuffle(perm)

    x = x[perm]
    y = y[perm]
    
    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
        
        yield batchX, batchY
        
# 定义性能指标函数
def mean(item):
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res

def genMetrics(trueY, predY, binaryPredY):
    ##predY表示的是预测的概率
    """
    生成acc和auc值
    """
    # auc = roc_auc_score(trueY,binaryPredY, multi_class="ovr", multi_class="ovr")
    # sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)；
    # normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY, average='weighted')
    recall = recall_score(trueY, binaryPredY, average='weighted')
    F = 2 * (precision * recall) / (precision + recall)
    return round(accuracy, 4), round(precision, 4), round(recall, 4),round(F, 4)
