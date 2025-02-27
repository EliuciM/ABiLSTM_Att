import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Configuration for Adversarial BiLSTM Neural Network with Bert Model')

    parser.add_argument("--seed", type=int, default=2021, help="random seed.2021")
    
    # ========================= Data Configs ==========================
    parser.add_argument('--data_path', type=str, default='./data/ctrip_20250224.csv')
    parser.add_argument('--base_url', type=str, default='./data/qunaer_20250226_balance19')           
    parser.add_argument('--num_class', type=int, default=3, help='2 for CSC, 15 for tnews, 119 for iflytek, 14 for THUCNews')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    # ========================= Training Configs ==========================
    parser.add_argument('--prefetch', default=12, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=6, type=int, help="num_workers for dataloaders")
    parser.add_argument('--train_batch_size', default=32, type=int, help="use for training duration per worker, 28 for tnews and THUCNews, 56 for CSC, 24 for BD and Tourism, 12 for Smart T3")
    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")
    
    # ========================= Word2Vec Configs ==========================
    parser.add_argument('--word2Vec', type=str, default='./word2vec/sgns.weibo.bigram-char.bz2')
    parser.add_argument('--word2Vec_dim', type=int, default=300)
    parser.add_argument('--stopWords', type=str, default='./stopword/hit_stopwords.txt')

    # ========================= BERT Configs ==========================
    parser.add_argument('--bert_dir', type=str, default='hf_hub/models--hfl--chinese-roberta-wwm-ext')
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_max_length', type=int, default=512)
    parser.add_argument('--bert_padding', type=str, default='max_length', help='args for tokenizer')
    # parser.add_argument('--bert_dropout',type=float, default=0.1, help='args for BertClassificationModel')
    
    # ========================= LSTM Configs ==========================
    parser.add_argument('--lstm_input_size', type=int, default=768, help='The number of expected features in the input x')
    parser.add_argument('--lstm_hidden_size', type=int, default=256, help='The number of features in the hidden state h')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--lstm_dropout',type=float, default=0.0)
    # parser.add_argument('--lstm_reflector_size', type=int, default=256, help='args for reflectors')
      
    # ========================= Fusion Configs ==========================
    parser.add_argument('--feature_type', type=int, default=1, choices=[1, 2])
    parser.add_argument('--fusion_type', type=str, default='cat', choices=['weighted', 'gate', 'residual', 'cat'], help='fusion type only work when feature_type is 2') 
    
    # ========================= Adversarial Configs ==========================
    parser.add_argument('--adv_type', type=str, default='none', choices=['fgm', 'pgd', 'freelb', 'smartp', 'none'])
    parser.add_argument('--fgm_adv_epsilon', type=float, default=1.0)
    
    parser.add_argument('--pgd_adv_epsilon', type=float, default=1.0)
    parser.add_argument('--pgd_adv_alpha', type=float, default=0.3)
    parser.add_argument('--pgd_adv_K', type=int, default=3)
    
    parser.add_argument('--freelb_adv_K', type=int, default=3)
    parser.add_argument('--freelb_adv_lr', type=float, default=1e-2)
    parser.add_argument('--freelb_adv_init_mag', type=float, default=2e-2)
    
    parser.add_argument('--smartp_adv_alpha', type=float, default=1.0)

    # ========================= Saved Model Configs ==========================
    parser.add_argument('--save_path', type=str, default='save', help='path to save model')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total steps to run')
    parser.add_argument('--warmup_steps', default=200, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--print_steps', type=int, default=10, help="Number of steps to log training metrics.")
    
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon of Adam optimizer for BertClassificationModel.")
    parser.add_argument('--adam_lr', default=5e-5, type=float, help='initial learning rate for BertClassificationModel')
    parser.add_argument('--dropout',type=float, default=0.1)

    parser.add_argument('--smoothing', default=0.0, type=float, help='coefficient for label smoothing')
    parser.add_argument('--patience', default=3, type=int, help='early stopping patience')

    return parser.parse_args()    