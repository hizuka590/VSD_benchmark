# coding: UTF-8
import time
import torch
import numpy as np
from utils.train_eval import train, init_network, summary
from importlib import import_module
import argparse
import yaml
from easydict import EasyDict
import os
from absl import app
from absl import flags, logging
from absl.logging import info
import json
from datetime import timedelta
import dataset.dataloader as VSD_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='hizuka s VSD')
parser.add_argument('--model', type=str, required=True, help='choose a model: cnn_ae, fully_ae, fast_text')
parser.add_argument('--yaml_file', type=str, default='/opt/sdb/polyu/VSD_benchmark/fully_ae.yaml', help='choose a yaml path')
parser.add_argument('--log_dir', type=str, default='logs', help='define a log path')


args = parser.parse_args()

def init_config(yaml_path):
    with open(args.yaml_file,'r') as f:
        config = yaml.safe_load(f)
    # 0. logging

    os.makedirs(args.log_dir+'/'+args.model, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {args.log_dir+'/'+args.model}")
    logging.get_absl_handler().use_absl_log_file()
    config["SAVE_DIR"] = args.log_dir+'/'+args.model
    config["MODEL"] = args.model

    # 2. info
    info(f"yaml config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    # test()
    # dataset = 'VSD'  # 数据集
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    x = import_module('models.' + model_name)

    # 保证每次结果一样
    config = init_config(args)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # load data
    start_time = time.time()
    print("Loading data...")
    train_dataset, valid_dataset = VSD_data.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=True)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    if config.MODEL == 'fully_ae':
        model = x.Model(config).to(config.DEVICE)
    if config.MODEL == 'cnn_ae':
        model = x.Model().to(config.DEVICE)

    # test
    init_network(model)
    if config.TEST_ONLY == True:
        print('summary only:')
        summary(config, model, train_loader, valid_loader)
    else:
        print('(train started ) model parameters: ', model.parameters)
        train(config, model, train_loader, valid_loader)


def test():
    print('============ test =============')
    print('args content: {}'.format(args))
    yaml_config = init_config(args)
    print('load yaml file: ',yaml_config)
    print('load yaml part: ', yaml_config.TRAIN_BATCH_SIZE)
    print('===============================')