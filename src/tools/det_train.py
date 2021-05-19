import cv2
import os
import yaml
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.utils_function import create_dir, create_module, create_loss_bin, save_checkpoint
from src.utils.logger import Logger
from src.utils.metrics import runningScore
from src.utils.cal_iou_acc import cal_DB
from src.utils.cal_recall_pre_f1 import cal_recall_precison_f1


GLOBAL_WORKER_ID = None
GLOBAL_SEED = 123456


def train_val_program(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    create_dir(config['base']['checkpoint'])
    model = create_module(config['model']['function'])(config)
    criterion = create_module(config['loss']['function'])(config)
    train_dataset = create_module(config['train_load']['function'])(config)
    val_dataset = create_module(config['val_load']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])(config)
    img_process = create_module(config['postprocess']['function'])(config)

    train_loader = DataLoader(train_dataset, batch_size=config['train_load']['batch_size'], shuffle=True,
                              num_workers=config['train_load']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['val_load']['batch_size'], shuffle=True,
                            num_workers=config['val_load']['num_workers'])
    loss_bin = create_loss_bin(config['base']['algorithm'])
    if torch.cuda.is_available():
        if len(config['base']['gpu_id'].split(',')) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion.cuda()

    start_epoch = 0
    recall, precision, hmean = 0, 0, 0
    best_recall, best_precision, best_hmean = 0, 0, 0
    if config['base']['restore']:
        print('Resume from checkpoint...')
        assert os.path.isfile(config['base']['restore_file']), 'checkpoint path is not correct'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_recall = checkpoint['recall']
        best_precision = checkpoint['precision']
        best_hmean = checkpoint['hmean']
        log_write = Logger(os.path.join(checkpoint, 'log.txt'), title=config['base']['althgorithm'], resume=True)
    else:
        print('Training from scratch...')
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    for epoch in range(start_epoch, config['base']['n_epoch']):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        loss_write = model_train(train_loader, model, optimizer, criterion, loss_bin, args, config, epoch)
        if epoch == config['base']['start_val']:
            create_dir(os.path.join(checkpoint, 'val'))
            create_dir(os.path.join(checkpoint, 'val', 'res_img'))
            create_dir(os.path.join(checkpoint, 'val', 'res_txt'))
            model.eval()
            recall, precision, hmean = model_eval(val_dataset, val_loader, model, img_process, checkpoint, config)
            print('recall:{:.4f} \tprecision:{:.4f} \t hmean:{:.4f}'.format(recall, precision, hmean))
            if hmean > best_hmean:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'hmean': hmean,
                    'precision': precision,
                    'recall': recall
                }, checkpoint=checkpoint, filename=config['base']['algorithm'] + '_best.pth')
                best_hmean = hmean
                best_precision = precision
                best_recall = recall
        for key in loss_bin.keys():
            loss_bin[key].loss_clear()
        if epoch % config['base']['save_epoch'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': config['optimizer']['base_lr'],
                'optimizer': optimizer.state_dict(),
                'hmean': 0,
                'precision': 0,
                'recall': 0,
            }, checkpoint=checkpoint, filename=config['base']['algorithm'] + '_best.pth')


def model_train(train_loader, model, optimizer, criterion, loss_bin, args, config, epoch):
    running_metric_text = runningScore(2)
    for batch_idx, data in enumerate(train_loader):
        pre_batch, gt_batch = model(data)
        loss, metrics = criterion(pre_batch, gt_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for key in loss_bin.keys():
            if key in metrics.keys():
                loss_bin[key].loss_add(metrics[key].item())
            else:
                loss_bin[key].loss_add(loss.item())
        iou, acc = cal_DB(pre_batch['binary'], pre_batch['gt'], pre_batch['mask'], running_metric_text)
        if batch_idx % config['base']['show_step'] == 0:
            log = '({}/{}/{}/{}) | ' \
                .format(epoch, config['base']['n_epoch'], batch_idx, len(train_loader))
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '

            log += 'ACC:{:.4f}'.format(acc) + ' | '
            log += 'IOU:{:.4f}'.format(iou) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            print(log)
    loss_write = []
    for key in list(loss_bin.keys()):
        loss_write.append(loss_bin[key].loss_mean())
    loss_write.extend([acc, iou])
    return loss_write


def model_eval(test_dataset, test_loader, model, imgprocess, checkpoint, config):
    for batch_idx, (imgs, ori_imgs) in enumerate(test_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        with torch.no_grad():
            out = model(imgs)
        scales = []
        if isinstance(out, dict):
            img_num = out['f_score'].shape[0]
        else:
            img_num = out.shape[0]
        for i in range(img_num):
            scale = (ori_imgs.shape[1] * 1.0 / out.shape[3], ori_imgs.shape[0] * 1.0 / out.shape[2])
            scales.append(scale)
        out = out.cpu().numpy()
        bbox_batch, score_batch = imgprocess(out, scales)
        for i in range(len(bbox_batch)):
            bboxes = bbox_batch[i]
            img_show = ori_imgs[i].numpy().copy()
            idx = i + out.shape[0] * batch_idx
            image_name = test_dataset.img_list[idx].split('/')[-1].split('.')[0]  # windows use \\ not /
            with open(os.path.join(checkpoint, 'val', 'res_txt', 'res_' + image_name + '.txt'), 'w+',
                      encoding='utf-8') as fid_res:
                for bbox in bboxes:
                    bbox = bbox.reshape(-1, 2).astype(np.int)
                    img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
                    bbox_str = [str(x) for x in bbox.reshape(-1)]
                    bbox_str = ','.join(bbox_str) + '\n'
                    fid_res.write(bbox_str)
            cv2.imwrite(os.path.join(checkpoint, 'val', 'res_img', image_name + '.jpg'), img_show)
    result_dict = cal_recall_precison_f1(config['val_load']['val_label_dir'], os.path.join(checkpoint, 'val', 'res_img'))
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--config', help='config path')
    # parameter training
    parser.add_argument('--num_epoch', type=int, default=1200, help='num of epoch training')
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--start_val', type=int, default=400)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--gpd_id', type=int, default=0)
    parser.add_argument()
    args = parser.parse_args()
    train_val_program(args)
