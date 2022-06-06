# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
from datetime import timedelta
from utils.helper_misc_tensor import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, cal_Jaccard, cal_BER

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter = 'None'):
    num_correct = 0
    num_pixels = 0
    start_time = time.time()
    # model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.SAVE_DIR + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            if config.MODEL =='fully_ae':
                trains = trains.view(-1, 416 * 416).to(config.DEVICE)
                labels = labels.data.reshape(trains.shape[0], -1).to(config.DEVICE)
            if config.MODEL =='cnn_ae' or config.MODEL == 'unet':
                # print('cnn/unet:===================')
                trains = trains.to(config.DEVICE)
                labels = labels.float().to(config.DEVICE)

            # print('input shape: ', trains.shape)
            # print('input shape: ', trains.shape)
            outputs = model(trains)
            outputs = outputs * 255.

            # print('output shape: ',outputs.shape)
            model.zero_grad()

            loss = nn.L1Loss(outputs, labels)
            # print(loss.item())
            loss.backward()
            optimizer.step()

            if total_batch % 200 == 0:
                print('start validation============')
                # 每多少轮输出在训练集和验证集上的效果
                # ===train set===
                # num_correct = 0
                # num_pixels = 0
                # dice_score = 0
                # train_acc = num_correct / num_pixels
                # predic = (outputs > 0.5).float().to(config.DEVICE)
                # num_correct += (predic == labels).sum()
                # num_pixels += torch.numel(predic)
                train_precision_record, train_recall_record, train_mae_record, train_Jaccard_record, train_BER_record, train_shadow_BER_record, train_non_shadow_BER_record = eval_metrics_init()
                outputs = outputs.to(device='cpu')
                labels = labels.to(device='cpu')
                print('number of classes in gt: ', set(np.array(labels.cpu()).flatten().tolist()))
                print('number of classes in predic: ', len(set(np.array(outputs.cpu()).flatten().tolist())),max(np.array(outputs.cpu()).flatten().tolist()))
                precision, recall, mae = cal_precision_recall_mae(outputs, labels)
                train_Jaccard_record.update(cal_Jaccard(outputs, labels))
                BER, shadow_BER, non_shadow_BER = cal_BER(outputs, labels)
                train_BER_record.update(BER)
                train_shadow_BER_record.update(shadow_BER)
                train_non_shadow_BER_record.update(non_shadow_BER)

                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    train_precision_record[pidx].update(p)
                    train_recall_record[pidx].update(r)
                train_mae_record.update(mae)

                fmeasure = cal_fmeasure([precord.avg for precord in train_Jaccard_record],
                                    [rrecord.avg for rrecord in train_recall_record])
                val_MAE,val_F_beta,val_Jaccard,val_BER,val_shadow_BER_record,val_non_shadow_BER_record= evaluate(config, model, dev_iter)
                # if dev_loss < dev_best_loss:
                #     dev_best_loss = dev_loss
                #     torch.save(model.state_dict(), config.save_path)
                #     improve = '*'
                #     last_improve = total_batch
                # else:
                #     improve = ''
                time_dif = get_time_dif(start_time)
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_acc, time_dif))
                writer.add_scalar("loss/train", loss.item(), total_batch)

                writer.add_scalar("MAE/train", train_mae_record.avg, total_batch)
                writer.add_scalar("J_beta/train", fmeasure, total_batch)
                writer.add_scalar("Jaccard/train", train_Jaccard_record.avg, total_batch)
                writer.add_scalar("BER/train", train_BER_record.avg, total_batch)
                writer.add_scalar("sBER/train", train_shadow_BER_record.avg, total_batch)
                writer.add_scalar("nsBER/train", train_non_shadow_BER_record.avg, total_batch)

                writer.add_scalar("MAE/validation", val_MAE, total_batch)
                writer.add_scalar("J_beta/validation", val_F_beta, total_batch)
                writer.add_scalar("Jaccard/validation", val_Jaccard, total_batch)
                writer.add_scalar("BER/validation", val_BER, total_batch)
                writer.add_scalar("sBER/validation", val_shadow_BER_record, total_batch)
                writer.add_scalar("nsBER/validation", val_non_shadow_BER_record.avg, total_batch)

                model.train()
            total_batch += 1
        #     if total_batch - last_improve > config.require_improvement:
        #         # 验证集loss超过1000batch没下降，结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break
        # if flag:

    writer.close()
    # test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    # num_correct = 0
    # num_pixels = 0
    # dice_score = 0
    precision_record, recall_record, mae_record, Jaccard_record, BER_record, shadow_BER_record, non_shadow_BER_record = eval_metrics_init()
    with torch.no_grad():
        for trains, labels in data_iter:
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            if config.MODEL =='fully_ae':
                trains = trains.view(-1, 416 * 416).to(config.DEVICE)
                labels = labels.data.reshape(trains.shape[0], -1).to(config.DEVICE)
            if config.MODEL =='cnn_ae' or config.MODEL == 'unet':
                trains = trains.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

            outputs = model(trains)
            predic = outputs * 255.
            predic = predic.to(device = 'cpu')
            labels = labels.to(device = 'cpu')
            # print('number of classes in gt: ', set(np.array(labels.cpu()).flatten().tolist()))
            # print('number of classes in predic: ', len(set(np.array(predic.cpu()).flatten().tolist())),
            #       max(np.array(predic.cpu()).flatten().tolist()))
            precision, recall, mae = cal_precision_recall_mae(predic, labels)
            Jaccard = cal_Jaccard(predic, labels)

            Jaccard_record.update(Jaccard)
            BER, shadow_BER, non_shadow_BER = cal_BER(predic, labels)
            BER_record.update(BER)
            shadow_BER_record.update(shadow_BER)
            non_shadow_BER_record.update(non_shadow_BER)

            for pidx, pdata in enumerate(zip(precision, recall)):
                p, r = pdata
                precision_record[pidx].update(p)
                recall_record[pidx].update(r)
            mae_record.update(mae)

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

        log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae_record.avg, fmeasure,
                                                                                       Jaccard_record.avg,
                                                                                       BER_record.avg,
                                                                                       shadow_BER_record.avg,
                                                                                       non_shadow_BER_record.avg)
        print(log)
        return mae_record.avg, fmeasure,Jaccard_record.avg,BER_record.avg,shadow_BER_record.avg,non_shadow_BER_record.avg



    #         predic = (outputs > 0.5).float().to(config.DEVICE)
    #         num_correct += (predic == labels).sum()
    #         num_pixels += torch.numel(predic)
    #         dice_score += (2*(predic*labels).sum()) / ((predic + labels).sum() )
    #
    #
    # acc = num_correct/num_pixels
    # print(f'Got {num_correct}/{num_pixels} with acc {acc:.2f}')
    # print(f'Dice score: {dice_score/len(data_iter)}')
    # return acc, dice_score/len(data_iter)



def eval_metrics_init():
    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()
    Jaccard_record = AvgMeter()
    BER_record = AvgMeter()
    shadow_BER_record = AvgMeter()
    non_shadow_BER_record = AvgMeter()
    return precision_record, recall_record, mae_record,Jaccard_record, BER_record, shadow_BER_record,non_shadow_BER_record


def summary(config, model, train_iter, dev_iter, test_iter = 'None'):
    from torchsummary import summary
    for i, (trains, labels) in enumerate(train_iter):
        print('train size :',trains.shape)
        print('label size : ',labels.shape,labels[0].view(-1).shape)
        #  trasnform below not work on get_item..
        labels[labels>0] = 1
        print('number of classes: ',set(labels[0].view(-1).tolist()))
        summary(model,trains.shape[1:])
        break
def evaluate_one(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:

            outputs = model(texts)
            print("question desc: ", texts)
            print("ground truth label: ", labels)
            print("predicted label : ", outputs)
if __name__ == "__main__":
    pass