# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from tensorboardX import SummaryWriter
from datetime import timedelta

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
            if config.MODEL =='fully_ae':
                trains = trains.view(-1, 416 * 416).to(config.DEVICE)
                labels = labels.data.reshape(trains.shape[0], -1).to(config.DEVICE)
            if config.MODEL =='cnn_ae':
                # print('cnn:===================')
                trains = trains.to(config.DEVICE)
                labels = labels.float().to(config.DEVICE)
            # print('input shape: ', trains.shape)
            # print('input shape: ', trains.shape)
            outputs = model(trains)

            # print('output shape: ',outputs.shape)
            model.zero_grad()
            loss = F.binary_cross_entropy(outputs, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()
            num_correct = 0
            if total_batch % 100 == 0:
                print('start validation============')
                # 每多少轮输出在训练集和验证集上的效果
                predic = (outputs > 0.5).float().to(config.DEVICE)
                num_correct += (predic == labels).sum()
                num_pixels += torch.numel(predic)
                train_acc = num_correct / num_pixels
                dev_acc, dev_dice = evaluate(config, model, dev_iter)
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
                # writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
        #     if total_batch - last_improve > config.require_improvement:
        #         # 验证集loss超过1000batch没下降，结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break
        # if flag:
        #     break
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
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for trains, labels in data_iter:
            if config.MODEL =='fully_ae':
                trains = trains.view(-1, 416 * 416).to(config.DEVICE)
                labels = labels.data.reshape(trains.shape[0], -1).to(config.DEVICE)
            if config.MODEL =='cnn_ae':
                trains = trains.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
            # print('input shape: ', trains.shape)
            # print('input shape: ', inputs.shape)
            # print('labels shape: ', labels.shape)
            outputs = model(trains)
            predic = (outputs > 0.5).float().to(config.DEVICE)
            num_correct += (predic == labels).sum()
            num_pixels += torch.numel(predic)
            dice_score += (2*(predic*labels).sum()) / ((predic + labels).sum() )


    acc = num_correct/num_pixels
    print(f'Got {num_correct}/{num_pixels} with acc {acc:.2f}')
    print(f'Dice score: {dice_score/len(data_iter)}')

    return acc, dice_score/len(data_iter)

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

def summary(config, model, train_iter, dev_iter, test_iter = 'None'):
    from torchsummary import summary
    for i, (trains, labels) in enumerate(train_iter):
        print('train size :',trains.shape)
        print('label size : ',labels.shape)
        summary(model,trains.shape[1:])
        break
