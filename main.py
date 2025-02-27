import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import time
import logging
import torch
import torch.multiprocessing as mp
from data_helper import create_dataloaders, load_dataloaders
from model import BertClassificationModel, BRNNAttClassifcationModel, BRNNAttClassifcationModelV2
from config import parse_args
import utils 
import matplotlib.pyplot as plt

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
    # 1. Load data
    train_dataloader, val_dataloader = load_dataloaders(args)
    num_total_steps = len(train_dataloader) * args.max_epochs
    args.max_steps = int(num_total_steps)
    args.warmup_steps = int(num_total_steps * 0.15)

    # 2. Build model and optimizers
    model = BRNNAttClassifcationModelV2(args)
    optimizer, scheduler = utils.build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. Adversarial training methods
    fgm = utils.FGM(model, epsilon=args.fgm_adv_epsilon, emb_name='word_embeddings.')
    pgd, K = utils.PGD(model, emb_name='word_embeddings.', epsilon=args.pgd_adv_epsilon, alpha=args.pgd_adv_alpha), args.pgd_adv_K
    freelb = utils.FreeLB(args.device, adv_K=args.freelb_adv_K, adv_lr=args.freelb_adv_lr, adv_init_mag=args.freelb_adv_init_mag)
    smart_adv, adv_alpha = utils.SmartPerturbation(args.device, loss_map={"0": torch.nn.functional.cross_entropy}), args.smartp_adv_alpha

    # 4. Training preparation
    step = 0
    best_f1 = args.best_score  # 记录最佳 f1-score
    best_loss = float('inf')   # 记录最佳验证 loss
    patience = args.patience   # 允许多少个 epoch 没有提升
    no_improve_epochs = 0      # 记录连续未提升的 epoch
    start_time = time.time()

    # 记录 loss 和评测指标
    train_losses, val_losses = [], []
    val_f1_scores = []
    steps_list = []

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_dataloader)

        for batch in train_dataloader:          
            if args.adv_type == 'fgm':
                loss, accuracy, pred_label, label, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                fgm.attack()
                loss_adv, _, _, _, _ = model(batch)
                loss_adv.backward()
                fgm.restore()
            
            elif args.adv_type == 'pgd':
                loss, accuracy, pred_label, label, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0))
                    if t != K-1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad()
                    
                    loss_adv, _, _, _, _ = model(batch)
                    loss_adv.backward()
                pgd.restore()               
            
            elif args.adv_type == 'freelb':
                loss, accuracy, pred_label, label = freelb.attack(model,batch)
                loss = loss.mean()
                accuracy = accuracy.mean()

            elif args.adv_type == 'smartp':
                loss_origin, accuracy, pred_label, label, logits = model(batch)
                loss_origin = loss_origin.mean()
                loss = loss_origin.clone()
                accuracy = accuracy.mean()
                loss_adv = smart_adv.forward(model,logits,batch)
                loss_origin = loss_origin + adv_alpha*loss_adv
                loss_origin.backward()

            else:
                loss, accuracy, pred_label, label, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            epoch_loss += loss.item()

            # 记录训练进度
            if step % args.print_steps == 0:
                avg_time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = avg_time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

                results_train = utils.evaluate(pred_label.cpu().numpy(), label.cpu().numpy())
                logging.info(f"Epoch {epoch}/{args.max_epochs} step {step}/{num_total_steps} eta {remaining_time}: loss {loss:.3f}, train {results_train}")

        # 记录平均训练损失
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # 5. Validation
        loss_val, results_val = validate(model, val_dataloader)
        val_losses.append(loss_val)

        results_val = {k: round(v, 4) for k, v in results_val.items()}
        val_f1_scores.append(results_val['f1'])
        steps_list.append(epoch)

        logging.info(f"Epoch {epoch} step {step}: loss {loss_val:.3f}, val {results_val}")

        # 6. Early Stopping: 检查 F1 是否提升 + Loss 是否下降
        if results_val['f1'] > best_f1:
            # 删除之前的模型
            if os.path.exists(f'{args.save_path}/model_best_f1_{best_f1:.4f}.bin'):
                os.remove(f'{args.save_path}/model_best_f1_{best_f1:.4f}.bin')

            best_f1 = results_val['f1']
            best_loss = loss_val
            no_improve_epochs = 0  # 重新计数未提升的 epoch

            # 保存最佳模型
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'f1': best_f1, 'loss': loss_val},
                       f'{args.save_path}/model_best_f1_{best_f1:.4f}.bin')
            

        elif loss_val > best_loss:  # F1 没变好且 Loss 上升
            no_improve_epochs += 1
            logging.info(f"Early Stopping Check: No improvement for {no_improve_epochs}/{patience} epochs")

            if no_improve_epochs >= patience:
                logging.info(f"Stopping early at epoch {epoch} due to no improvement in validation F1 and increasing loss.")
                break

    # 7. 绘制 loss 和评测指标曲线
    plot_metrics(steps_list, train_losses, val_losses, val_f1_scores, args.save_path)

def plot_metrics(steps, train_losses, val_losses, val_f1_scores, save_path):
    """绘制训练损失、验证损失和 f1-score 变化曲线"""
    plt.figure(figsize=(10, 5))

    # 训练 & 验证 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label="Train Loss", marker="o")
    plt.plot(steps, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()

    # f1-score 曲线
    plt.subplot(1, 2, 2)
    plt.plot(steps, val_f1_scores, label="Val F1-score", marker="o", color="r")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.legend()
    plt.grid()

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_metrics.png")

def main():
    args = parse_args()
    utils.setup_logging()
    utils.setup_device(args)
    utils.setup_seed(args)

    save_name = f"feature{args.feature_type}_fusion{args.fusion_type}_adv{args.adv_type}_{time.strftime('%Y%m%d%H%M%S')}"
    args.save_path = os.path.join(args.save_path, os.path.basename(args.base_url), os.path.basename(args.bert_dir), save_name)
    os.makedirs(args.save_path)
    
    fh = logging.FileHandler(os.path.join(args.save_path, 'train.log'), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    logging.info("Training/evaluation config: %s", args)

    train_and_validate(args)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    main()