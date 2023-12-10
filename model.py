import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, AutoModel

class FARNNAttClassificationModel(nn.Module):
    def __init__(self, args, pretrained_embeddings):
        super().__init__()
        self.args = args
        self.cls = args.num_class
        self.device = args.device

        self.word_embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1]).from_pretrained(pretrained_embeddings, freeze=False)

        self.bilstm = nn.LSTM(args.lstm_input_size, args.lstm_hidden_size, num_layers=args.lstm_num_layers,
                                batch_first=args.lstm_batch_first, dropout=args.lstm_dropout, 
                                bidirectional=args.lstm_bidirectional).to(args.device)    
        
        self.attention_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size),
            nn.ReLU(inplace=True)
        ).to(args.device) 

        self.reflector = nn.Linear(args.lstm_hidden_size*2 + args.word2Vec_dim, args.lstm_reflector_size).to(self.device)

        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.lstm_hidden_size, self.cls)
        ).to(args.device) 

    def forward(self, inputs:dict(), inference=False):
        '''
        :param inputs['input_ids']:    [batch_size, len_seq] 
        :param inputs['label']:        [batch_size, 1]
        '''
        # w2v_embedding [batch_size, len_seq, word2Vec_dim]
        w2v_embeddings = self.word_embeddings(inputs['input_ids'])
        
        # h_0, c_0 [num_layers*num_directions, batch_size, hidden_size]
        h_0 = torch.randn(self.args.lstm_num_layers*2, self.args.train_batch_size, self.args.lstm_hidden_size).to(self.device)
        c_0 = torch.randn(self.args.lstm_num_layers*2, self.args.train_batch_size, self.args.lstm_hidden_size).to(self.device)

        # final_hidden_state 对应 h_n, final_cell_state 对应 c_n, 形状同 [num_layers*num_directions, batch_size, hidden_size]
        # output [batch_size, len_seq, hidden_size*num_directions]
        output, (final_hidden_state, final_cell_state) = self.bilstm(w2v_embeddings)

        # final_hidden_state: [batch_size, num_layers*num_directions, hidden_size]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)

        # lstm_output: [[batch_size, len_seq, hidden_size], [batch_size, len_seq, hidden_size]]
        lstm_output = torch.chunk(output, 2, -1) # 分成前后两块, fw_output 以及 bw_output
        
        # att_out: [batch_size, hidden_size]
        att_out = self.attention(lstm_output, final_hidden_state)

        # word_repre [batch_size, len_seq, hidden_size *2 + word2Vec_dim]
        word_repre = torch.cat((lstm_output[0], w2v_embeddings, lstm_output[1]), dim=-1)

        # text_repre [batch_size, len_seq, hidden_size]
        text_repre = self.reflector(word_repre)

        # text_repre [batch_size, hidden_size]
        text_repre, _ = torch.max(text_repre, dim=1)

        prediction_att_out = self.classifier(att_out)

        prediction_text_repre = self.classifier(text_repre)

        prediction = (prediction_att_out + prediction_text_repre)/2

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])
    
    def cal_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label   

    def attention(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [[batch_size, len_seq, hidden_size], [batch_size, len_seq, hidden_size]]
        :param lstm_hidden: [batch_size, num_layers * num_directions, hidden_size]
        :return:            [batch_size, hidden_size]
        '''
        # h [batch_size, time_step, hidden_size] time_step我觉得和 len_seq是一致的，这个地方存个疑问
        h = lstm_out[0] + lstm_out[1]

        # lstm_hidden [batch_size, hidden_size]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)

        # lstm_hidden [batch_size, 1, hidden_size]
        lstm_hidden = lstm_hidden.unsqueeze(1)

        # att_w [batch_size, 1, hidden_size]
        att_w = self.attention_layer(lstm_hidden)

        # m [batch_size, time_step, hidden_size]
        m = nn.Tanh()(h)

        # att_context [batch_size, 1, time_step]
        att_context = torch.bmm(att_w, m.transpose(1, 2))
        
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(att_context, dim=-1)
        
        # context [batch_size, 1, hidden_size]
        context = torch.bmm(softmax_w, h)
        
        # result [batch_size, hidden_size]
        result = context.squeeze(1)
        
        return result

class BRNNAttClassifcationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cls = args.num_class
        self.device = args.device 
        self.bert = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, output_hidden_states = True).to(args.device)

        self.bilstm = nn.LSTM(args.lstm_input_size, args.lstm_hidden_size, num_layers=args.lstm_num_layers,
                        batch_first=args.lstm_batch_first, dropout=args.lstm_dropout, 
                        bidirectional=args.lstm_bidirectional).to(args.device)    
        
        self.bilstm_attention_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size),
            nn.ReLU(inplace=True)
        ).to(args.device) 
        
        # self.Singleclassifier = nn.Linear(args.bert_dim, self.cls).to(args.device)

        self.Multiclassifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim + args.lstm_hidden_size*2, args.bert_dim + args.lstm_hidden_size*2),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim + args.lstm_hidden_size*2, self.cls)
        ).to(args.device)

    def forward(self, inputs:dict, inference=False):

        if 'inputs_embeds' in inputs:
            bert_output = self.bert(inputs_embeds = inputs['inputs_embeds'], attention_mask = inputs['attention_mask'], token_type_ids = inputs['token_type_ids'])
        else:
            bert_output = self.bert(inputs['input_ids'], attention_mask = inputs['attention_mask'], token_type_ids = inputs['token_type_ids'])

        last_hidden_state = bert_output['last_hidden_state']

        # CLS_feature = last_hidden_state[:,0,:]
        
        # Word_feature = last_hidden_state[:,1:-1,:] # 取出最后一层每个字对应的特征向量作为 字向量 (batch,seq,feature)

        # final_hidden_state 对应 h_n, final_cell_state 对应 c_n, 形状同 [num_layers*num_directions, batch_size, hidden_size]
        # output [batch_size, len_seq, hidden_size*num_directions]
        output, (final_hidden_state, _) = self.bilstm(last_hidden_state)

        # final_hidden_state: [batch_size, num_layers*num_directions, hidden_size]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)

        # lstm_output: [[batch_size, len_seq, hidden_size], [batch_size, len_seq, hidden_size]]
        lstm_output = torch.chunk(output, 2, -1) # 分成前后两块, fw_output 以及 bw_output
        
        Pooler = bert_output['pooler_output']
        
        # afw/abw/att [batch_size, hidden_size]
        afw = self.attention(lstm_output[0], final_hidden_state)

        abw = self.attention(lstm_output[1], final_hidden_state)

        # att = self.attention(lstm_output[0] + lstm_output[1], final_hidden_state)

        # cat [batch_size, len_seq, hidden_size *2 + bert_dim]
        # cat = torch.cat((lstm_output[0], last_hidden_state, lstm_output[1]), dim=-1)

        # catout [batch_size, hidden_size *2 + bert_dim]   
        # catout, _ = torch.max(cat, dim=1)

        # afw_Pooler_abw [batch_size, hidden_size *2 + bert_dim]
        afw_Pooler_abw = torch.cat((afw, Pooler, abw), dim=-1)

        # fusion = torch.cat((afw, abw), dim = -1)

        prediction = self.Multiclassifier(afw_Pooler_abw)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    def attention(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, hidden_size]
        :param lstm_hidden: [batch_size, num_layers * num_directions, hidden_size]
        :return:            [batch_size, hidden_size]
        '''
        # h [batch_size, time_step, hidden_size] time_step 我觉得和 len_seq是一致的，这个地方存个疑问
        h = lstm_out

        # lstm_hidden [batch_size, hidden_size]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)

        # lstm_hidden [batch_size, 1, hidden_size]
        lstm_hidden = lstm_hidden.unsqueeze(1)

        # att_w [batch_size, 1, hidden_size]
        att_w = self.bilstm_attention_layer(lstm_hidden)

        # m [batch_size, time_step, hidden_size]
        m = nn.Tanh()(h)

        # att_context [batch_size, 1, time_step]
        att_context = torch.bmm(att_w, m.transpose(1, 2))
        
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(att_context, dim=-1)
        
        # context [batch_size, 1, hidden_size]
        context = torch.bmm(softmax_w, h)
        
        # result [batch_size, hidden_size]
        result = context.squeeze(1)
        
        return result

    def cal_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        if self.args.smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.zeros_like(prediction).to(self.device)
                true_dist.fill_(self.args.smoothing / (self.args.num_class - 1))
                true_dist.scatter_(1, label.data.unsqueeze(1), 1 - self.args.smoothing)
            KL_loss = nn.KLDivLoss(reduction = 'batchmean')
            loss = KL_loss(F.log_softmax(prediction, dim=1), true_dist)
        
        elif self.args.smoothing == 0.0:
            loss = F.cross_entropy(prediction, label)
        
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        
        return loss, accuracy, pred_label_id, label, prediction  

class BertClassificationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cls = args.num_class
        self.device = args.device 
        self.bert = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, output_hidden_states = True).to(args.device)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, use_fast=True)
        self.bert_classifier = nn.Linear(args.bert_dim, self.cls).to(args.device)  #bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, inputs:dict, inference=False):
        for key, tensor in inputs.items():
            # 检查tensor是否在model的设备上
            if tensor.device != self.bert.device:
                # 如果不在，将tensor移动到model的设备上
                inputs[key] = tensor.to(self.bert.device)

        if 'inputs_embeds' in inputs:
            bert_output = self.bert(inputs_embeds = inputs['inputs_embeds'], attention_mask = inputs['attention_mask'], token_type_ids = inputs['token_type_ids'])
        else:
            bert_output = self.bert(inputs['input_ids'], attention_mask = inputs['attention_mask'], token_type_ids = inputs['token_type_ids'])

        Pooler_feature = bert_output['pooler_output']

        # hidden_states = bert_output['hidden_states']

        # last_hidden_state = bert_output['last_hidden_state']

        # CLS_feature = last_hidden_state[:,0,:]
        
        # # Word_feature = last_hidden_state[:,1:-1,:] # 取出最后一层每个字对应的特征向量作为 字向量 (batch,seq,feature)

        # # [batch_size, 4*bert_dim]
        # CLS_last3_features = torch.cat((hidden_states[-1][:,0,:],hidden_states[-2][:,0,:],hidden_states[-3][:,0,:],Pooler_feature), dim=-1)

        # # [batch_szie, 3*bert_dim]
        # CLS_last2_features = torch.cat((hidden_states[-1][:,0,:],hidden_states[-2][:,0,:],Pooler_feature), dim=-1)

        # # [batch_szie, 2*bert_dim]
        # CLS_last_features = torch.cat((hidden_states[-1][:,0,:],Pooler_feature), dim=-1)

        prediction2 = self.bert_classifier(Pooler_feature)

        prediction = prediction2

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    def cal_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        if self.args.smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.zeros_like(prediction).to(self.device)
                true_dist.fill_(self.args.smoothing / (self.args.num_class - 1))
                true_dist.scatter_(1, label.data.unsqueeze(1), 1 - self.args.smoothing)
            KL_loss = nn.KLDivLoss(reduction = 'batchmean')
            loss = KL_loss(F.log_softmax(prediction, dim=1), true_dist)
        
        elif self.args.smoothing == 0.0:
            loss = F.cross_entropy(prediction, label)
        
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        
        return loss, accuracy, pred_label_id, label, prediction

