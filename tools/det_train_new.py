import cv2
from tqdm import tqdm
import os
import yaml
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
from src.utils.utils_function import create_dir, create_module, save_checkpoint
from src.utils.logger import Logger
from src.utils.metrics import runningScore, cal_text_score, QuadMetric
warnings.filterwarnings('ignore')


def train_val_program(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['base']['gpu_id']

    create_dir(config['base']['checkpoint'])
    checkpoints_path = config['base']['checkpoint']

    model = create_module(config['model']['function'])(config)
    criterion = create_module(config['model']['loss_function'])(config)
    train_dataset = create_module(config['train_load']['function'])(config)
    val_dataset = create_module(config['val_load']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model.parameters())
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    img_process = create_module(config['postprocess']['function'])(config)
    metric_cls = QuadMetric()
    train_loader = DataLoader(train_dataset, batch_size=config['train_load']['batch_size'], shuffle=True,
                              num_workers=config['train_load']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['val_load']['batch_size'], shuffle=False,
                            num_workers=config['val_load']['num_workers'])

    if torch.cuda.is_available():
        if len(config['base']['gpu_id'].split(',')) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    start_epoch = 1
    best_hmean = 0
    if config['base']['restore']:
        print('Resume from checkpoint...')
        assert os.path.isfile(config['base']['restore_file']), 'checkpoint path is not correct'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_hmean = checkpoint['hmean']
    else:
        print('Training from scratch...')
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    for epoch in range(start_epoch, config['base']['n_epoch'] + 1):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        train_loss = model_train(train_loader, model, criterion, optimizer, config, epoch)
        print('Train loss:', train_loss)
        model.eval()
        recall, precision, hmean = model_eval(val_loader, model, img_process, config, metric_cls)
        print('Recall:{}, precision:{}, hmean:{}'.format(recall, precision, hmean))
        # save per epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': config['optimizer']['base_lr'],
            'optimizer': optimizer.state_dict(),
        }, checkpoints_path, filename=config['base']['algorithm'] + '_current.pth')
        if hmean > best_hmean:
            best_hmean = hmean
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': config['optimizer']['base_lr'],
                'optimizer': optimizer.state_dict(),
            }, checkpoints_path, filename=config['base']['algorithm'] + '_best.pth')


def model_train(train_loader, model, criterion, optimizer, config, epoch):
    running_metric_text = runningScore(2)
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        lr = optimizer.param_groups[0]['lr']
        preds = model(data[0])
        assert preds.size(1) == 3
        _batch = torch.stack([data[1], data[2], data[3], data[4]])
        total_loss = criterion(preds, _batch)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        score_shrink_map = cal_text_score(preds[:, 0, :, :],
                                          data[1], data[2], running_metric_text, thresh=0.3)
        acc = score_shrink_map['Mean Acc']
        iou_shrink_map = score_shrink_map['Mean IoU']
        train_loss += total_loss
        if batch_idx % config['base']['show_step'] == 0:
            print('Epoch:{} - Step:{} - lr:{} - loss:{} - acc:{} - iou:{}'.format(epoch+1, batch_idx, lr, total_loss,
                                                                              acc, iou_shrink_map))
    end_epoch_loss = train_loss / len(train_loader)
    return end_epoch_loss


def model_eval(test_loader, model, imgprocess, config, metric_cls):
    raw_metrics = []
    for idx, test_batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            test_preds = model(test_batch[0])
            assert test_preds.size(1) == 2
            batch_shape = {'shape': config['base']['crop_shape']}
            box_list, score_list = imgprocess(batch_shape, test_preds,
                                              is_output_polygon=config['postprocess']['is_poly'])
            raw_metric = metric_cls.validate_measure(test_batch, (box_list, score_list))
            raw_metrics.append(raw_metric)
    metrics = metric_cls.gather_measure(raw_metrics)
    recall = metrics['recall'].avg
    precision = metrics['precision'].avg
    hmean = metrics['hmean'].avg
    return recall, precision, hmean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--config', help='config path')
    parser.add_argument('--start_epoch', type=int, default=None)
    # parameter training
    parser.add_argument('--start_val', type=int, default=1)
    args = parser.parse_args()
    train_val_program(args)
