import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import os
import yaml
import argparse
import warnings
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
from src.utils.utils_function import create_dir, create_module, save_checkpoint, dict_to_device
from src.utils.det_metrics import runningScore, cal_text_score, QuadMetric
from src.logger.logger import setup_logging
warnings.filterwarnings('ignore')


def get_data_loader(cfg, logger):
    train_dataset = create_module(cfg['dataset']['function'])(cfg, is_training=True)
    test_dataset = create_module(cfg['dataset']['function'])(cfg, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['train_load']['batch_size'],
                              shuffle=True, num_workers=cfg['dataset']['train_load']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['dataset']['test_load']['batch_size'],
                             shuffle=True, num_workers=cfg['dataset']['test_load']['num_workers'])
    logger.info('Loaded successful!. Train datasets: {}, test datasets: {}'.format(len(train_dataset), len(test_dataset)))
    return train_loader, test_loader


def get_logger(name):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    return logger


def main(args):
    with open(args.config, 'r') as stream:
        cfg = yaml.safe_load(stream)
    # create log and save file
    save_dir = Path(cfg['base']['save_dir'])
    exp_name = cfg['base']['algorithm']
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_model_dir = save_dir / 'ckpt' / exp_name / run_id
    log_dir = save_dir / 'logs' / exp_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)
    logger = get_logger('train')

    # create train, test loader
    train_loader, test_loader = get_data_loader(cfg, logger)

    model = create_module(cfg['model']['function'])(cfg)
    logger.info('Model created, trainable parameters:')
    criterion = create_module(cfg['loss']['function'])(cfg['loss']['l1_scale'], cfg['loss']['bce_scale'])
    optimizer = create_module(cfg['optimizer']['function'])(cfg, model.parameters())
    post_process = create_module(cfg['post_process']['function'])(cfg)
    logger.info('Optimizer created.')
    logger.info('Training start...')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['base']['gpu_id']
    if torch.cuda.is_available():
        if len(cfg['train']['gpu_id'].split(',')) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    if args.resume:
        assert os.path.isfile(cfg['base']['ckpt_file']), 'checkpoint path is not correct'
        logger.info('Resume from checkpoint: {}'.format(cfg['base']['ckpt_file']))
        checkpoint = torch.load(cfg['base']['ckpt_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_hmean = checkpoint['hmean']
    else:
        logger.info('Training from scratch...')
    """
    start_epoch = 1
    best_hmean = 0
    metric_cls = QuadMetric()
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    for epoch in range(start_epoch, config['base']['n_epoch'] + 1):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        train_loss = model_train(train_loader, model, criterion, optimizer, config, epoch)
        print('Train loss:', train_loss.item())
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
    """


def model_train(train_loader, model, criterion, optimizer, config, epoch):
    running_metric_text = runningScore(2)
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        lr = optimizer.param_groups[0]['lr']
        data = dict_to_device(data)
        preds = model(data['img'])
        assert preds.size(1) == 3
        _batch = torch.stack([data['gt'], data['gt_mask'], data['thresh_map'], data['thresh_mask']])
        total_loss = criterion(preds, _batch)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        score_shrink_map = cal_text_score(preds[:, 0, :, :],
                                          data['gt'], data['gt_mask'], running_metric_text, thresh=0.3)
        acc = score_shrink_map['Mean Acc']
        iou_shrink_map = score_shrink_map['Mean IoU']
        train_loss += total_loss
        if batch_idx % config['base']['show_step'] == 0:
            print('Epoch:{} - Step:{} - lr:{} - loss:{} - acc:{} - iou:{}'.format(epoch, batch_idx, lr, total_loss,
                                                                              acc, iou_shrink_map))
    end_epoch_loss = train_loss / len(train_loader)
    return end_epoch_loss


def model_eval(test_loader, model, imgprocess, config, metric_cls):
    raw_metrics = []
    for idx, test_batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            test_data = dict_to_device(test_data)
            test_preds = model(test_data['img'])
            assert test_preds.size(1) == 2
            batch_shape = {'shape': [(736, 736)]}
            box_list, score_list = imgprocess(batch_shape, test_preds,
                                              is_output_polygon=config['postprocess']['is_poly'])
            raw_metric = metric_cls.validate_measure(test_batch, (box_list, score_list))
            raw_metrics.append(raw_metric)
    metrics = metric_cls.gather_measure(raw_metrics)
    recall = metrics['recall'].avg
    precision = metrics['precision'].avg
    hmean = metrics['fmeasure'].avg
    return recall, precision, hmean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--config', type=str, default='config/db_resnet50.yaml', help='config path')
    parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
    args = parser.parse_args()
    main(args)

