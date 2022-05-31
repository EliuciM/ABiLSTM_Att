 
import tensorflow.compat.v1 as tf

# 加入对抗以及注意力机制的双向长短期记忆网络，二分类
class ABiLSTM_Attention_2class(object):
    tf.compat.v1.enable_eager_execution()
    def __init__(self, config, wordEmbedding, indexFreqs):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        
        # 根据词的频率计算权重
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)
        
        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用词频计算新的词嵌入矩阵
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)
            
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)
            
         # 计算二元交叉熵损失 
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._ABiLSTM_Attention(self.embeddedWords)
                self.binaryPreds = tf.cast(tf.greater_equal(self.predictions,0.0), tf.float32, name="binaryPreds")
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
                loss = tf.reduce_mean(losses)
        
        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._ABiLSTM_Attention(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)
        
        self.loss = loss + perturLoss
            
    def _ABiLSTM_Attention(self, embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """
        config = self.config
        
        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
           
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)


            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                          self.embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm")
        
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            
            # 逐位相加[batch_size, max_time, hidden_size]
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)

            outputSize = config.model.hiddenSizes
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.keras.initializers.glorot_normal())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB") #创建一个数值为0.1的常量
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            
        return predictions

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hiddenSizes
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]，tf.reshape(M, [-1, hiddenSize])
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        # tf.transpose[H,[0,2,1]]，代表第一个维度不变，把第一维度里面的矩阵转置，结果[batch_size,hidden_size,time_step]
        # tf.reshape(self.alpha, [-1, config.sequenceLength, 1]),相当于[batch_size,time_step,1]
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))
        
        # r.shape=[batch_size,hidden_size,1]
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output.shape=[batch_size, hidden_size]
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output
        
    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        mean = tf.matmul(weights, wordEmbedding)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        var = tf.matmul(weights, powWordEmbedding)
        stddev = tf.sqrt(1e-6 + var)
        return (wordEmbedding - mean) / stddev

    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        ##保证不训grad
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
            
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

# 不加注意力，BiLSTM后直接一个全连接层，二分类
class ABiLSTM_2class(object):
    tf.compat.v1.enable_eager_execution()
    def __init__(self, config, wordEmbedding, indexFreqs):
        
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        
        # 根据词的频率计算权重
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)
        
        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用词频计算新的词嵌入矩阵
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)
            
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)
            
         # 计算二元交叉熵损失 
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._ABiLSTM(self.embeddedWords)
                self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
                loss = tf.reduce_mean(losses)
        
        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._ABiLSTM(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)
        
        self.loss = loss + perturLoss
            
    def _ABiLSTM(self, embeddedWords):
        """
        Bi-LSTM
        """
        config = self.config
        
        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
           
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)


            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                          self.embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm")
        self.embeddedWords_ = tf.concat(outputs, 2) #高维拼接，把hidden_size层所在的括号去掉，然后合并里面的数值，相当于两倍的hidden_size长度

        # 去除最后时间步的输出作为全连接的输入
        finalOutput = self.embeddedWords_[:, -1, :]
        
        outputSize = config.model.hiddenSizes * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.keras.initializers.glorot_normal())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB") #创建一个数值为0.1的常量
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            
        return predictions
        
    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        #相当于已经求了平均
        mean = tf.matmul(weights, wordEmbedding)

        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        #表示的随机变量的方差
        var = tf.matmul(weights, powWordEmbedding)

        stddev = tf.sqrt(1e-6 + var)
        
        return (wordEmbedding - mean) / stddev

    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        ##保证不训grad
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
            
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit


class ABiLSTM_Attention_3class(object):
    tf.compat.v1.enable_eager_execution()
    def __init__(self, config, wordEmbedding, indexFreqs):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        
        # 根据词的频率计算权重
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)
        
        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用词频计算新的词嵌入矩阵，wordEmbedding是一个词和词向量的转换矩阵，行是词编码
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)
            
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)
            
         # 计算二元交叉熵损失 
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._Bi_LSTMAttention(self.embeddedWords)
                binaryPreds1 = tf.cast(tf.greater_equal(self.predictions, 0.55), tf.float32, name="binaryPreds1")
                binaryPreds1=binaryPreds1*2
                temp1 = tf.cast(tf.greater(self.predictions, 0.45), tf.float32, name="binaryPreds2")
                temp2 = tf.cast(tf.less(self.predictions, 0.55), tf.float32, name="binaryPreds2")
                temp1=temp1-2
                binaryPreds2=tf.add(temp1,temp2,name=None)#求中性
                self.binaryPreds=tf.add(binaryPreds1,binaryPreds2,name=None)
                #这一步相当于把预测概率值位于[0.45,0.55]区间范围内的数据判断为0，把小于0.45的数值看为-1，把大于0.55的数值看为1，相当于设置了三分类
                                
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
#                 losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=self.inputY)#多分类
                loss = tf.reduce_mean(losses)
        
        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
#                 perturLosses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)
        
        self.loss = loss + perturLoss
            
    def _Bi_LSTMAttention(self, embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """
        config = self.config
        
        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
           
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)


            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                          self.embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm")

        
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            outputSize = config.model.hiddenSizes
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.keras.initializers.glorot_normal())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            
        return predictions
    
    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hiddenSizes
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        return output
    
    #对词嵌入层做标准化处理的一个函数，不影响网络的结构
    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        #相当于已经求了平均，因为weights本身就是一个标准化后的行向量，
        #出现的次数越多，贡献越大
        mean = tf.matmul(weights, wordEmbedding)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        ##表示的随机变量的方差
        var = tf.matmul(weights, powWordEmbedding)
        stddev = tf.sqrt(1e-6 + var)
        
        #z-score法（正规化方法）CSDN搜索: 数据标准化的三种最常用方式总结（归一化）
        return (wordEmbedding - mean) / stddev
    
    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        """
        ADDN：全部计算好后，在调用ADD_N操作计算
        EXPERIMENTAL_ACCUMULATE_N：这个是计算一个累加一个，应该可以减少内存占用
        EXPERIMENTAL_TREE：这个方法具体什么算法不知道，但文档说降低性能，能提高内存的利用率，而且未来可能不再支持。
        """
        
        ##保证不训grad
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        
        return embedded + perturb
    
    def _scaleL2(self, x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit


class ABiLSTM_3class(object):
    tf.compat.v1.enable_eager_execution()
    def __init__(self, config, wordEmbedding, indexFreqs):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        
        # 根据词的频率计算权重，cast为转换类型的函数，indexFreqs记录了每一个词的出现次数，行是词的编码，数量是N，列是某一编码词出现的次数，数量是1，转换成1行N列
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)
        
        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用词频计算新的词嵌入矩阵，wordEmbedding是一个词和词向量的转换矩阵，行是词编码
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)
            
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)
            
         # 计算二元交叉熵损失 
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._Bi_LSTMAttention(self.embeddedWords)
                binaryPreds1 = tf.cast(tf.greater_equal(self.predictions, 0.55), tf.float32, name="binaryPreds1")
                binaryPreds1=binaryPreds1*2
                temp1 = tf.cast(tf.greater(self.predictions, 0.45), tf.float32, name="binaryPreds2")
                temp2 = tf.cast(tf.less(self.predictions, 0.55), tf.float32, name="binaryPreds2")
                temp1=temp1-2
                binaryPreds2=tf.add(temp1,temp2,name=None)#求中性
                self.binaryPreds=tf.add(binaryPreds1,binaryPreds2,name=None)
                #这一步相当于把预测概率值位于[0.45,0.55]区间范围内的数据判断为0，把小于0.45的数值看为-1，把大于0.55的数值看为1，相当于设置了三分类
                                
                #这个地方用inputY是对的，和MNIST视频中介绍的方法是一致的，具体的细节可以看看代价函数那一章
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
#                 losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=self.inputY)#多分类
                loss = tf.reduce_mean(losses)
        
        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
#                 perturLosses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)
        
        self.loss = loss + perturLoss
            
    def _Bi_LSTMAttention(self, embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """
        config = self.config
        
        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
           
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                                                         output_keep_prob=self.dropoutKeepProb)


            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                          self.embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm")
        self.embeddedWords_ = tf.concat(outputs, 2) #高维拼接，把hidden_size层所在的括号去掉，然后合并里面的数值，相当于两倍的hidden_size长度
        
        # 去除最后时间步的输出作为全连接的输入
        finalOutput = self.embeddedWords_[:, -1, :]
        
        outputSize = config.model.hiddenSizes * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.keras.initializers.glorot_normal())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            
        return predictions
    
    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hiddenSizes
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        return output
    
    #对词嵌入层做标准化处理的一个函数，不影响网络的结构
    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        #相当于已经求了平均，因为weights本身就是一个标准化后的行向量，（1*N）*（N*300）=1*300，相当于得到了一个所有词在三百个维度上数值的平均值，但是考虑了词出现的次数对于最终数值的影响，即
        #出现的次数越多，贡献越大
        mean = tf.matmul(weights, wordEmbedding)
        print(mean)
        #（N*300)-（1*300）得到一个（x-x拔）²，也是N*300维度
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        ##表示的随机变量的方差
        var = tf.matmul(weights, powWordEmbedding)
        print(var)
        stddev = tf.sqrt(1e-6 + var)
        
        #z-score法（正规化方法）CSDN搜索: 数据标准化的三种最常用方式总结（归一化）
        return (wordEmbedding - mean) / stddev
    
    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        """
        ADDN：全部计算好后，在调用ADD_N操作计算
        EXPERIMENTAL_ACCUMULATE_N：这个是计算一个累加一个，应该可以减少内存占用
        EXPERIMENTAL_TREE：这个方法具体什么算法不知道，但文档说降低性能，能提高内存的利用率，而且未来可能不再支持。
        """
        
        ##保证不训grad
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        
        return embedded + perturb
    
    def _scaleL2(self, x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit