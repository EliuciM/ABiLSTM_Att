import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for Adversarial BiLSTM Neural Network with Bert Model')

    parser.add_argument("--seed", type=int, default=2021, help="random seed.2021")
    parser.add_argument('--dropout',type=float, default=0.1)
    # ========================= Data Configs ==========================
    parser.add_argument('--data_path', type=str, default='./data/BD.csv')
    parser.add_argument('--base_url', type=str, default='./data/')
    # parser.add_argument('--file_type', type=str, default='json')            
    parser.add_argument('--num_class', type=int, default=3, help='2 for CSC, 15 for tnews, 119 for iflytek, 14 for THUCNews')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    parser.add_argument('--prefetch', default=12, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=6, type=int, help="num_workers for dataloaders")

    parser.add_argument('--train_batch_size', default=32, type=int, help="use for training duration per worker, 28 for tnews and THUCNews, 56 for CSC, 24 for BD and Tourism, 12 for Smart T3")
    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")

    parser.add_argument('--word2Vec', type=str, default='./word2vec/sgns.weibo.bigram-char.bz2')
    parser.add_argument('--word2Vec_dim', type=int, default=300)
    parser.add_argument('--stopWords', type=str, default='./stopword/hit_stopwords.txt')

    # ========================= BERT Configs ==========================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='./model/hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_dim', type=int, default=768)
    
    parser.add_argument('--bert_max_length', type=int, default=512, help='args for tokenizer')
    parser.add_argument('--bert_padding', type=str, default='max_length', help='args for tokenizer')
    
    # parser.add_argument('--bert_dropout',type=float, default=0.1, help='args for BertClassificationModel')
    
    # ========================= LSTM Configs ==========================
    parser.add_argument('--lstm_input_size', type=int, default=768, help='The number of expected features in the input x')
    
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='The number of features in the hidden state h')
    
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--lstm_batch_first', type=bool, default=True, help='the input and output tensors are in shape of (batch,seq,feature)')
    parser.add_argument('--lstm_dropout',type=float, default=0.0)
    parser.add_argument('--lstm_bidirectional',type=bool ,default=True)
      
    # Official API document: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM

    parser.add_argument('--lstm_reflector_size', type=int, default=256, help='args for reflectors')
    
    # ========================= Saved Model Configs ==========================
    parser.add_argument('--savedmodel_path', type=str, default='save/BD/v13')
    parser.add_argument('--ckpt_file', type=str, default='save/BD/v13/model_.bin')
    parser.add_argument('--log_file', type=str, default='log/BD/v13.log')

    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--warmup_steps', default=200, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--print_steps', type=int, default=10, help="Number of steps to log training metrics.")
    
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon of Adam optimizer for BertClassificationModel.")
    parser.add_argument('--adam_lr', default=5e-5, type=float, help='initial learning rate for BertClassificationModel')

    parser.add_argument('--smoothing', default=0.0, type=float, help='coefficient for label smoothing')

    return parser.parse_args()    