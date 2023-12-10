### T3中的结果默认batch=24,seed=2021
### 原始的V7代表了12，即afw+Pooler+abw
## 日志文件的命名规则：v13_ernie-1.0-base-zh_h256_b12_s2022
- v13：代表论文中提到的1，3特征的组合
- ernie-1.0-base-zh：预训练模型名称
- h256：LSTM的hiddensize大小
- b12：batchsize为12
- s2021：seed为2021.
### 1、Test: 验证模型构造函数中冗余部分对于分类结果的影响
- v1_ernie-1.0-base-zh_h256_b12_s2021_temp2 是使用BertClassificationModel跑出来的结果，foward中存在一些其它的东西的
- v1_ernie-1.0-base-zh_h256_b12_s2021_temp 是使用BRNNAttClassifcationModel跑出来的，分类器使用了Singleclassifier
- v1_ernie-1.0-base-zh_h256_b12_s2021 是使用BRNNAttClassifcationModel跑出来的，最后的分类器使用了Multiclassifier
- v1_ernie-1.0-base-zh_h256_b12_s2021_temp3 BertClassificationModel跑出来的结果，foward只有用到的东西
- v1_ernie-1.0-base-zh_h256_b12_s2021_temp4 BRNNAttClassifcationModel跑出来的，去除了构造函数中未用到的东西，和temp3是一致的，这一点说明构造函数中的模型组成会影响最后的结果。
### 2、Test: 验证seed以及batch对于分类结果的影响
- T3_AutoModel文件夹是使用AutoModel加载预训练模型，
- 其中进行了seed还有batch参数的测试（最好使用dev测试集独立出来可以比较好的衡量模型的效果。）
### 3、Test: 验证预训练语言模型对于分类结果的影响
- 需要选定一种特征级联方法来判断预训练模型的结果，需要考虑单独的Pooler特征v1和加入后续结构v13,v12对结果的影响。
- 同时也需要考虑超参数hiddensize的影响，这需要在选好级联方法之后再选。
### 4、Test: 验证HiddenSize对模型结果的影响
### 5、Test: 验证不同特征级联方法对于模型结构的影响
### 6、Test：验证模型在两两不同的标签上的分类精度情况
