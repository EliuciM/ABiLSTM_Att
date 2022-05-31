from gensim.models import KeyedVectors
import pandas as pd
import jieba
import re
import numpy as np
from collections import Counter

# 正则匹配所有的非中文词 
sub_re='[^\u4e00-\u9fa5]'

class Dataset(object):
    def __init__(self, config, PathAll):
        self._dataSource = PathAll.dataSource
        self._stopWordSource = PathAll.stopWordList  
        
        self._sequenceLength = config.sequenceLength    # 每条输入的序列处理为定长,所有评论词语数量的均值加两倍的标准差
        self._embeddingSize = config.model.embeddingSize    # 词向量的维度
        self._batchSize = config.batchSize  
        self._rate = config.rate    # 训练集的比例，0说明训练集为0
        
        self._stopWordDict = {}
        self._reviewId=[]   # 记录评论的id序号，方便后续的查询处理

        self.trainId=[]
        self.trainReviews = []  # 训练语料分词后对应的序号
        self.trainLabels = []   # 训练集的标签  
        self.train_ReviewsText=[]   # 训练语料原文

        self.testId=[]
        self.testReviews=[] # 测试语料分词后对应的序号
        self.testLabels = []    # 测试集的标签
        self.test_ReviewsText=[]    # 测试语料原文
        
        self.indexFreqs = []  # 统计词空间中的词在出现在多少个review中
        
        self._wordToIndex = {}
        self._indexToWord = {}
        self.allWords =[]

        self.wordEmbedding = None
        self.wordVec = KeyedVectors.load_word2vec_format(PathAll.word2vec,binary=False)

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """  
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["reviewtext"].tolist()
        reviewId = df["reviewid"].tolist()
        #这里需要分词
        reviews=[]
        for line in review:#每次读一行
            line = re.sub(sub_re, '', line)
            #结巴分词
            cut_text=jieba.lcut(line)
            cut_list=[i for i in cut_text]
            reviews.append(cut_list)
        return reviewId, reviews, labels
    
    def _readText(self, filePath):
        """
        从csv文件中读取数据集
        这里没有分词
        """  
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["reviewtext"].tolist()
        #这里不需要分词
        reviewsText=[]
        for line in review:#每次读一行
            reviewsText.append(line)
        return reviewsText, labels

    def _reviewProcess(self, review,sequenceLength):#将文本序列化
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        UNK对应的索引为1
        """
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength 
        # 判断当前的序列是否小于定义的固定序列长度
                
        if len(review) < sequenceLength:
            sequenceLen = len(review)
            
        for i in range(sequenceLen):
            #这就相当于过滤掉停用和低频词之后的结果生成词向量
            if review[i] in self._wordToIndex:
                #reviewVec[i] = self.wordVec.vocab[review[i]].index
                reviewVec[i] = self._wordToIndex[review[i]]
            else:
                reviewVec[i] = 0    
        return reviewVec
    
    def _getWordEmbedding(self, words):

        vocab = []
        wordEmbedding = []
        for word in words:
            if  word in self.wordVec:
                #在已经训练好的词向量中进行查询
                vector = self.wordVec[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            else:
                #如果不就随机生成一个UNK的初始词向量
                vocab.append(word)
                #wordEmbedding.append(tf.random_uniform([1,self._embeddingSize], -1.0, 1.0))
                wordEmbedding.append(np.random.randn(self._embeddingSize))
                #print(word + "不存在于词向量中") 

        return vocab, np.array(wordEmbedding)
    
    def _getWordIndexFreq(self, vocab, reviews):
        """
        统计词汇空间中各个词出现在多少个文本中
        """
        reviewDicts = [dict(zip(review, range(len(review)))) for review in reviews]
        indexFreqs = [0] * len(vocab)
        for word in vocab:
            count = 0
            for review in reviewDicts:
                if word in review:
                    count += 1
            indexFreqs[self._wordToIndex[word]] = count
        
        self.indexFreqs = indexFreqs
        
    def _readStopWord(self, stopWordList):
        """
        读取停用词
        """
        with open(stopWordList, "r",encoding="utf-8") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
    
    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        在没有词嵌入的时候可以用的到
        """
        
        allWords = [word for review in reviews for word in review]#所有词
        
        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]
        
        wordCount = Counter(subWords)  # 统计词频字典，如{'a': 2, 'c': 1, 'b': 1}
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)#降序，字典
        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >=0]#如果词频大约0则返回词，词频的大小设置有待考虑
        
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        #vocab这里应该与wordEmbedding生成的词嵌入顺序一一对应
        
        #用gensim这种索引都可以不用.
        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))
        self.allWords=words
        # 得到逆词频
        self._getWordIndexFreq(vocab, reviews)
           
    def _genTrainTestData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        labels = []
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            ##每一个review对应一个label
            reviewVec = self._reviewProcess(x[i],self._sequenceLength)#将文本序列化，不等长
            reviews.append(reviewVec)
            labels.append([y[i]])

        reviews = np.asarray(reviews, dtype="int64") #reviews里面是序号

        # 超出词嵌入大小的常用词词向量用0代替
        num_words=self.wordEmbedding.shape[0]
        reviews[reviews>=num_words ] = 0

        trainIndex = int(len(x) * rate)
        
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        
        testReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        testLabels = np.array(labels[trainIndex:], dtype="float32")
        
        return trainReviews, trainLabels, testReviews, testLabels

    def _genTrainTestText(self, x, rate):
        """
        生成训练集和验证集文本
        没有把文本转化成数字化表示
        """
        reviews = []
        labels = []
        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(x[:trainIndex])
        testReviews = np.asarray(x[trainIndex:])
        return trainReviews,testReviews    

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化停用词
        self._readStopWord(self._stopWordSource) 
        # 初始化数据集
        reviewId, reviews, labels = self._readData(self._dataSource)  # 完成分词之后的reviews
        reviewsText, labelsText = self._readText(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)
        # 记录评论序列号
        self._reviewId = reviewId

        trainIndex = int(len(self._reviewId)*self._rate)
        self.trainId = np.array(reviewId[:trainIndex], dtype="float32")
        self.testId = np.array(reviewId[trainIndex:], dtype="float32")
        # 初始化训练集和测试集
        trainReviews, trainLabels, testReviews, testLabels = self._genTrainTestData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.testReviews = testReviews
        self.testLabels = testLabels
        #生成训练集和验证集文本
        train_ReviewsText, test_ReviewsText = self._genTrainTestText(reviewsText, self._rate)
        self.train_ReviewsText = train_ReviewsText
        self.test_ReviewsText = test_ReviewsText    
        