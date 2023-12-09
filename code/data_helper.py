import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__))+ '/transformers/src')

import pandas as pd
import numpy as np
from functools import partial
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from datasets import load_dataset
from transformers import BertTokenizer
from gensim.models import KeyedVectors
from collections import Counter
# import jieba
import re
import json
from sklearn.preprocessing import LabelEncoder

def create_dataloaders(args, dataseType: str = 'BERT'):
    if dataseType == 'BERT':
        dataset = BERTDataset(args, path = args.data_path)
        # dataset.args.bert_max_length = dataset.seqLen if dataset.seqLen < args.bert_max_length else args.bert_max_length # 将输入字的数量调整为较小
    elif dataseType == 'W2V':
        dataset = W2VDataset(args, args.data_path)
    
    size = len(dataset)
    val_size = int(size*args.val_ratio)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                    generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader

def load_dataloaders(args):
    assert args.base_url != ''
    base_url = args.base_url
    # raw_datasets = load_dataset('json', data_files={'train': base_url + 'train.json', 'test': base_url + 'test.json', 'dev': base_url + 'dev.json'})
    
    train_data = []
    for f in open(base_url+'train.json', 'r', encoding='utf8'):
        train_data.append(json.loads(f))
    val_data = []
    for f in open(base_url+'dev.json', 'r', encoding='utf8'):
        val_data.append(json.loads(f))
    
    text_list = [i['sentence'] for i in train_data]
    label_list = [i['label'] for i in train_data]
    label_list = LabelEncoder().fit_transform(label_list)
    train_dataset = BERTDataset(args, text_list = text_list, label_list = label_list)

    text_list = [i['sentence'] for i in val_data]
    label_list = [i['label'] for i in val_data]
    label_list = LabelEncoder().fit_transform(label_list)
    val_dataset = BERTDataset(args, text_list = text_list, label_list = label_list)
   
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)        
    return train_dataloader, val_dataloader

class BERTDataset(Dataset):
    def __init__(self, args, path:str = '', text_list:list = [], label_list:list = [], test_mode:bool = False):
        self.args = args
        self.test_mode = test_mode
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, use_fast=True)

        assert len(text_list) == len(label_list)
        
        if len(text_list) != 0 and len(label_list) != 0:
            self.text = text_list
            self.label = label_list
        else:
            df = pd.read_csv(path)
            self.text = df["reviewtext"].tolist()
            self.label = df["sentiment"].tolist()
        
        self.seqLen = self.get_sequenceLength(self.text)

    def encode_bert_input(self, text: list):
        encoded_inputs = self.tokenizer(text, max_length=self.args.bert_max_length, padding=self.args.bert_padding, truncation=True)
        
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        attention_mask = torch.LongTensor(encoded_inputs['attention_mask'])
        token_type_ids = torch.LongTensor(encoded_inputs['token_type_ids'])

        return input_ids, attention_mask, token_type_ids

    def get_sequenceLength(self, text:list):
        seq_list = [len(str(i)) for i in text]
        mean_length = np.mean(seq_list)
        seq_length = mean_length + 2*np.std(seq_list) + 2 # add 2 tokens [CLS] [SEP]
        return int(seq_length)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> dict:
        input_ids, attention_mask, token_type_ids = self.encode_bert_input(self.text[index])
        data = dict(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        if not self.test_mode:
            data['label'] = torch.LongTensor([self.label[index]])

        return data

class W2VDataset(Dataset):
    def __init__(self, args, path:str, test_mode:bool = False):
        self.args = args
        self.test_mode = test_mode
        self.wordVec = KeyedVectors.load_word2vec_format(args.word2Vec,binary=False)
        self.stopWords = self.get_stopWordList(args.stopWords)
        self.sub_re = '[^\u4e00-\u9fa5]' # 正则匹配所有的中文词
        
        df = pd.read_csv(path)
        self.text = df["reviewtext"].tolist()
        self.label = df["sentiment"].tolist()

        self.cutText = self.get_cutTextList()
        self.seqLen = self.get_sequenceLength(self.cutText) # 得到长度适中的文本的词语数量

        self.wordProcess(self.cutText)
    
    def get_stopWordList(self, path:str):
        with open(path, "r",encoding="utf-8") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
        return stopWordList

    def get_cutTextList(self):
        cutTextList = []
        for line in self.text: # 每次读一行
            line = re.sub(self.sub_re, '', line) # 去除所有非中文词语
            cut_text=jieba.lcut(line) # 结巴分词
            cut_list=[i for i in cut_text if i not in self.stopWords] # 去除在停用词列表中的词语
            cutTextList.append(cut_list)
        return cutTextList

    def get_sequenceLength(self, text:list):
        seq_list = [len(i) for i in text]
        mean_length = np.mean(seq_list)
        seq_length = mean_length + 2*np.std(seq_list)
        return int(seq_length)

    def wordProcess(self, text:list): # 生成词向量表，词频表，词与序列的互查字典
        wordCount = {}
        for line in text:
            line = np.unique(line) # 去重, 即同一个句子中出现多次的词语不计
            for word in line:
                if word not in wordCount:
                    wordCount[word] = 1
                    continue
                wordCount[word] = wordCount[word] + 1

        # allWords = [word for line in text for word in line] # 所有词
        # wordCount = Counter(subWords)  # 统计词频字典，如{'a': 2, 'c': 1, 'b': 1}
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True) # 降序，字典
        # 如果词频大于1则返回词，词频的大小设置有待考虑
        vocab = [item[0] for item in sortWordCount if item[1] > 1]
        
        wordEmbedding = [] # 词语的词向量表
        wordFrequency = [] # 词语出现在所有评论中的频率
        wordSummation = sum(wordCount.values()) # 所有词语出现次数的和
        for word in vocab:
            # 在已经训练好的词向量中进行查询, 如果不存在就随机生成一个初始词向量, ( 不是停用词，但也不在词向量中 )
            vector = self.wordVec[word] if word in self.wordVec else np.random.randn(self.args.word2Vec_dim)
            wordEmbedding.append(vector)
            wordFrequency.append(wordCount[word]/wordSummation)
        
        weight = np.reshape(wordFrequency,(1,-1)) # [1, wordNum]
        mean = np.matmul(weight, wordEmbedding) # [1, W2VDim] = [1, wordNum] * [wordNum, W2VDim] 

        pow_wordEmbedding = np.power(wordEmbedding - mean, 2) # [wordNum, W2VDim]
        var = np.matmul(weight, pow_wordEmbedding) # [1, W2VDim] = [1, wordNum] * [wordNum, W2VDim]
        stddev = np.sqrt(1e-6 + var)    

        normEmbedding = (wordEmbedding - mean) / stddev # [wordNum, W2VDim]

        self.vocab = vocab
        # print([item[0] for item in sortWordCount if item[1] > 0 and item[1]<2])
        self.wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        # self.indexToWord = dict(zip(list(range(len(vocab))), vocab))
        self.wordEmbedding = torch.Tensor(np.array(normEmbedding))
        # self.wordEmbedding = normEmbedding
        # self.wordFrequency = wordFrequency

    def encode_w2v_input(self, text:list): # 根据指定长度补齐review，并将其序列化
        input_ids = []
        # 对于超出文本长度的空词语应该用什么补需要考虑一下，这里使用某一个不重要的字对应的向量来补齐
        for i in range(self.seqLen):
            index = self.wordToIndex[text[i]] if i < len(text) and text[i] in self.wordToIndex else len(self.vocab)-1
            # vector = self.wordEmbedding[index]
            input_ids.append(index)    
        return torch.LongTensor(np.array(input_ids))
        # return torch.Tensor(np.array(input_ids))         
    
    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> dict:
        input_ids = self.encode_w2v_input(self.cutText[index])
        data = dict( input_ids = input_ids )

        if not self.test_mode:
            data['label'] = torch.LongTensor([self.label[index]])

        return data