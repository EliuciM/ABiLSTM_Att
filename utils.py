import random
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, BertModel

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def show_bertmodel(args):
    model = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, output_hidden_states = True)
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def build_optimizer(args, model):
    # emb_name 为了保证词嵌入层不参与参数的更新，但是在optimizer.step的时候可以参与梯度的计算，这样就可以增加对抗样本了
    no_decay = ['bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.adam_lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=args.max_steps)
    return optimizer, scheduler

def evaluate(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    acc = accuracy_score(labels, predictions)
    result = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return result

class FGM():
    '''
    Example
    # 初始化
    fgm = utils.FGM(model,epsilon=1,emb_name='word_embeddings.')
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            # 对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            loss_adv, _, _, _, _ = model(batch)
            loss_adv.backward() # 反向传播,并在正常的grad基础上,累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    '''
    def __init__(self, model,emb_name,epsilon=0.5):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm!=0 and not torch.isnan(norm):
                    #对梯度进行scale，然后乘以一个epsilon系数，得到对抗样本
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    '''
    Example
    pgd, K = utils.PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3), 3
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()
                
                loss_adv, _, _, _, _ = model(batch)
                loss_adv.backward() # 反向传播, 并在正常的grad基础上, 累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    '''
 
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
 
    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
 
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
 
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
 
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class FreeLB(object):
    """
    Example
    freelb = FreeLB(args.device,adv_K=3,adv_lr=1e-2,adv_init_mag=2e-2)
        for batch in train_dataloader:
            model.train()
            loss, accuracy = freelb.attack(model,batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    """
 
    def __init__(self, device , adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        """
        初始化
        :param adv_K: 每次扰动对抗的小步数,最少是1 一般是3
        :param adv_lr: 扰动的学习率1e-2
        :param adv_init_mag: 初始扰动的参数 2e-2
        :param adv_max_norm:0  set to 0 to be unlimited 扰动的大小限制 torch.clamp()等来实现
        :param adv_norm_type: ["l2", "linf"]
        :param base_model: 默认的bert
        """
        self.device = device
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag    # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
 
    def attack(self, model, inputs, gradient_accumulation_steps=1):
        # model 可以放在初始化中
 
        input_ids = inputs['input_ids'].to(self.device)
 
        # labels = inputs['label'].squeeze(dim=1).to(self.device)
        
        # 得到初始化的embedding
        # 从bert模型中拿出embeddings层中的word_embeddings来进行input_ids到embedding的变换
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings(input_ids)
 
        if self.adv_init_mag > 0:   # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            #类型和设备转换
            input_mask = inputs['attention_mask'].to(self.device)
            input_lengths = torch.sum(input_mask, 1)
            
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化
 
        for astep in range(self.adv_K):
            
            delta.requires_grad_()
            # bert transformer类模型在输入的时候inputs_embeds 和 input_ids 二选一 不然会报错。。。。。。源码
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            inputs['input_ids'] = None
 
            # 下游任务的模型，我这里在模型输出没有给出loss 要自己计算原始loss
            loss, accuracy, _, _, _ = model(inputs)
            accuracy = accuracy.mean()
            loss = loss/self.adv_K
 
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
 
            if astep == self.adv_K - 1:
                # further updates on delta
                break
 
            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad
 
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings(input_ids) 
        
        return loss, accuracy

class SmartPerturbation():
    """
    step_size noise扰动学习率
    epsilon 梯度scale时防止分母为0
    norm_p 梯度scale采用的范式
    noise_var 扰动初始化系数
    loss_map 字典,loss函数的类型{"0":mse(),....}
    使用方法
    optimizer =
    model =
    smart_adv = SmartPerturbation(args.device, loss_map = {"0":torch.nn.functional.cross_entropy})
    adv_alpha = 1
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _, logits = model(batch)
            accuracy = accuracy.mean()
            loss_adv = smart_adv.forward(model,logits,batch)
            loss = loss + adv_alpha*loss_adv
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    """
    def __init__(self,
                 device,
                 base_model='bert',
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 loss_map={},
                 norm_level=0):
        super(SmartPerturbation, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.base_model = base_model
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0
 
    # 梯度scale
    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction
 
    # 初始noise扰动
    def generate_noise(self,embed, mask, epsilon=1e-5):
        noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
        noise.detach()
        noise.requires_grad_()
        return noise
 
    # 对称散度loss
    def stable_kl(self, logit, target, epsilon=1e-6, reduce=True):
        logit = logit.view(-1, logit.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()
 
    # 对抗loss输出
    def forward(self, model, logits, inputs, task_id='0', task_type="Classification", pairwise=1):
        # adv training
        assert task_type in set(['Classification', 'Ranking', 'Regression']), 'Donot support {} yet'.format(task_type)
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        target = inputs['label'].to(self.device)

        # init delta
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings(input_ids)
        
        # embed生成noise
        noise = self.generate_noise(embeds_init, attention_mask, epsilon=self.noise_var)
        
        # noise更新K轮
        for step in range(0, self.K):
            inputs['inputs_embeds'] = embeds_init + noise
            # noise + embed得到对抗样本的输出logits
            _, _, _, _, adv_logits = model(inputs)
            # adv_logits = torch.autograd.Variable(pred_label_id.float(), requires_grad = True)
            if task_type == 'Regression':
                adv_loss = F.mse_loss(adv_logits, target, logits.detach(), reduction='sum')
            else:
                if task_type == 'Ranking':
                    adv_logits = adv_logits.view(-1, pairwise)
                adv_loss = self.stable_kl(adv_logits, target, logits.detach(), reduce=False)
 
            # 得到noise的梯度
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            
            # 得到新的scale的noise
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        
        inputs['inputs_embeds'] = embeds_init + noise
        
        _, _, _, _, adv_logits = model(inputs)
        
        if task_type == 'Ranking':
            adv_logits = adv_logits.view(-1, pairwise)
        
        adv_lc = self.loss_map[task_id]
        
        # 计算对抗样本的对抗损失
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        # adv_loss = self.stable_kl(logits, adv_logits)
        
        return adv_loss
 
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑Loss
    """
    def __init__(self, classes, smoothing=0.1, dim=-1, device='cuda'):
        """
        :param classes: 类别数目
        :param smoothing: 平滑系数
        :param dim: loss计算平均值的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.device = device
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()
 
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred).to(self.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # torch.mean(torch.sum(-true_dist * pred, dim=self.dim))就是按照公式来计算损失
        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        # 采用KLDivLoss来计算
        loss = self.loss(pred,true_dist)
        return loss
