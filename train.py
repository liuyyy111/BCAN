# -*- coding:UTF-8 -*-

# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Training script"""

import os
import time
import shutil

import torch
import torch.nn as nn
import numpy
from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
import argparse
import numpy as np
import random
from data import get_loaders
from vocab import deserialize_vocab
from model import SCAN, ContrastiveLoss
from evaluation import AverageMeter, encode_data, LogCollector, i2t, t2i, shard_xattn

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed) #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速


def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()


def main():
    setup_seed(1024)
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='D:/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_epochs', default=15, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='./runs/test2',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/test2',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--correct_type', default="prob",
                        help='equal|prob')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true', default=True,
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_softmax', default=20., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--gpuid', default=0, type=str,
                        help='gpuid')
    opt = parser.parse_known_args()[0]
    print(opt.gpuid)
    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)

    torch.autograd.set_detect_anomaly(True)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('train')
    # tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt)
    model.cuda()
    model = nn.DataParallel(model)

    criterion = ContrastiveLoss(margin=opt.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Eiters is used to show logs as the continuation of another
            # training
            # model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        message = "epoch: %d, model name: %s\n" % (epoch, opt.model_name)
        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, message)

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, criterion, optimizer, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            # 'Eiters': model.Eiters,
            'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict()
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.images, self.captions, self.length, self.index = next(self.loader)
        except StopIteration:
            self.images, self.captions, self.length, self.index = None, None, None, None
            return
        with torch.cuda.stream(self.stream):
            self.images = self.images.cuda()
            self.captions = self.captions.cuda()
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preload()
        return self.images, self.captions, self.length, self.index


def train(opt, train_loader, model, criterion, optimizer, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    run_time = 0
    start_time = time.time()
    prefetcher = DataPrefetcher(train_loader)
    images, captions, lengths, index = prefetcher.next()
    i = 0
    while images is not None:
        # switch to train mode
        model.train()
        # measure data loading time


        # make sure train logger is used
        model.logger = train_logger

        optimizer.zero_grad()
        # Update the model
        if torch.cuda.device_count() > 1:
            images = images.repeat(torch.cuda.device_count(), 1, 1)
        score = model(images, captions, lengths, index)
        loss = criterion(score)
        loss.backward()

        # scaler.scale(loss).backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
            # scaled_loss.backward()

        if opt.grad_clip > 0:
          clip_grad_norm_(model.parameters(), opt.grad_clip)
            # clip_grad_norm_(amp.master_params(optimizer), opt.grad_clip)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        # measure elapsed time

        if (i + 1) % opt.log_step == 0:
            run_time += time.time() - start_time
            log = "epoch: %d; batch: %d/%d; loss: %.4f; time: %.4f" % (epoch,
                                                                       i, len(train_loader), loss.data.item(),
                                                                       run_time)
            print(log, flush=True)
            start_time = time.time()
            run_time = 0
        # Record logs in tensorboard
        # tb_logger.log_value('epoch', epoch, step=model.Eiters)
        # tb_logger.log_value('step', i, step=model.Eiters)
        # tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        # tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        # model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        images, captions, lengths, index = prefetcher.next()
        i += 1

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, img_means, cap_embs, cap_lens, cap_means = encode_data(
        model, val_loader, opt.log_step, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = shard_xattn(model, img_embs, img_means, cap_embs, cap_lens, cap_means, opt, shard_size=128)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    # tb_logger.log_value('r1', r1, step=model.Eiters)
    # tb_logger.log_value('r5', r5, step=model.Eiters)
    # tb_logger.log_value('r10', r10, step=model.Eiters)
    # tb_logger.log_value('medr', medr, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr, step=model.Eiters)
    # tb_logger.log_value('r1i', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i, step=model.Eiters)
    # tb_logger.log_value('medri', medri, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                message = "--------save best model at epoch %d---------\n" % (state["epoch"] - 1)
                print(message, flush=True)
                log_file = os.path.join(prefix, "performance.log")
                logging_func(log_file, message)
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
