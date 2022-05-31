# ABiLSTM_Att
融合对抗与注意力机制的Bi-LSTM网络在景区评论情感分析中的应用

### 情感分类流程
- 0、数据标注，规范语句结构
- 1、word2vec词向量训练
- 2、使用标注语句进行模型训练
- 3、使用未标注语句进行情感预测
- 4、结合预测值生成评价分数

### 环境配置
- Python 3.8.12
- Tensorflow 2.3.0
- Gensim 4.1.2
- Jieba 0.42.0
<!-- 代码是在tensorflow2.0环境下运行的，但使用到了tf1.0的API，若使用tf1.0环境运行则需要修改相关引用 -->

### 数据目录
数据目录结构如下：
```bash
.
├── data # 语料
│   ├── 2class
|   |    ├── JD_MixedTrain.csv
|   |    ├── reviews_wuhan_labeled_plus.csv 
|   |    └── reviews_wuhan_labeled.csv
│   ├── 3class
|   |     ├── reviews_wuhan_labeled_plus.csv
|   |     └── reviews_wuhan_labeled.csv
│   ├── corpus
|   |     └──reviews_wuhan_seg.txt
|   └── result
|
├── logs # 日志文件
|
├── model_load # 预训练模型路径
│   ├── xxxxx
│   └── xxxxx
│
├── model_save # 训练模型保存路径
│   ├── xxxxx
│   └── xxxxx
│
├── stopwords # 停用词
│   └── hit_stopwords.txt   # https://github.com/goto456/stopwords
│
├── word2vec
│   ├── sgns.weibo.bigram-char
│   │   └── sgns.weibo.bigram-char # https://github.com/Embedding/Chinese-Word-Vectors
│   ├── gensim_word2vec.ipynb
│   └── xxxx
│
├── config.py
├── dataSet.py
├── main_load.ipynb
├── main_save.ipynb
├── model.py
├── README.md
└── tools.py
```

### 代码介绍
- [config.py](config.py) 是配置文件
- [dataSet.py](dataSet.py) 是数据预处理模块
- [tools.py](tools.py) 是计算序列长度以及评价指标的函数工具
- [model.py](model.py) 是baseline模型
- [main_load.ipynb](main_load.ipynb) 加载预训练模型进行结果预测
- [main_save.ipynb](main_save.ipynb) 读取训练数据进行结果预测
