import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import logging
import torch
import torch.multiprocessing as mp
from data_helper import create_dataloaders, load_dataloaders
from model import BertClassificationModel, FARNNAttClassificationModel, BRNNAttClassifcationModel
from config import parse_args
import utils 

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label, _ = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = utils.evaluate(predictions, labels)

    model.train()
    return loss, results

def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    num_total_steps = len(train_dataloader) * args.max_epochs
    args.max_steps = int(num_total_steps)
    args.warmup_steps = int(num_total_steps * 0.15)

    # 2. build model and optimizers
    model = BertClassificationModel(args)
    optimizer, scheduler = utils.build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    
    fgm = utils.FGM(model,epsilon=1,emb_name='word_embeddings.')
    pgd, K = utils.PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3), 3    
    freelb = utils.FreeLB(args.device,adv_K=3,adv_lr=1e-2,adv_init_mag=2e-2)
    smart_adv, adv_alpha = utils.SmartPerturbation(args.device, loss_map = {"0":torch.nn.functional.cross_entropy}), 0.75

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_dataloader:
            loss, accuracy, _, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            # if step % args.print_steps == 0:
            #     time_per_step = (time.time() - start_time) / max(1, step)
            #     remaining_time = time_per_step * (num_total_steps - step)
            #     remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            #     logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")


        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")


        # 5. save checkpoint
        f1 = results['f1']
        if f1 > best_score:
            best_score = f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'f1': f1, 'loss': loss},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_f1_{f1}_loss_{loss:.3f}.bin')

def main():
    args = parse_args()
    utils.setup_logging()
    utils.setup_device(args)
    utils.setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # 创建一个FileHandler实例，模式为 'w'
    fh = logging.FileHandler(args.log_file, 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    logging.info("Training/evaluation config: %s", args)

    train_and_validate(args)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()