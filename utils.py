import os
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)


def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)


def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def get_lr_scheduler(lr_scheduler, optimizer, args):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.25)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def kd_loss(student_outputs, labels, teacher_outputs, alpha, temp):
    """Knowledge Distillation Loss"""
    KD_loss = F.kl_div(F.log_softmax(student_outputs/temp, dim=1),
                       F.softmax(teacher_outputs/temp, dim=1),
                       reduction='batchmean')
    CE_loss = F.cross_entropy(student_outputs, labels)
    total_loss = KD_loss * (1.0 - alpha) * (temp ** 2) + CE_loss * alpha
    return total_loss


def plot_losses(train_losses, val_losses, args):
    """Plot Losses"""
    plt.figure(1)
    plt.plot(train_losses, label='Train Loss', alpha=0.5)
    plt.plot(val_losses, label='Val Loss', alpha=0.5)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    if args.tto:
        plt.title('Student Model')
    elif not args.kd:
        plt.title('Without Knowledge Distillation Loss')
    elif args.kd:
        plt.title('Knowledge Distillation Loss')
    
    plt.grid()
    plt.legend(loc='best')
    
    if args.tto:
        plt.savefig(os.path.join(args.plots_path, 'Teacher Model Loss using {} {} {}.png'.format(args.teacher_type, args.dataset, args.num_classes)))
    elif not args.kd:
        plt.savefig(os.path.join(args.plots_path, 'Without Knowledge Distillation Loss using {} {}.png'.format(args.dataset, args.num_classes)))
    elif args.kd:
        plt.savefig(os.path.join(args.plots_path, 'Knowledge Distillation Loss using {} {} at temp {}.png'.format(args.dataset, args.num_classes, args.temp)))
    else:
        raise NotImplementedError


def plot_accuracies(train_top1_accuracies, train_top5_accuracies, val_top1_accuracies, val_top5_accuracies, args):
    """Plot Accuracies"""
    plt.figure(2)
    plt.plot(train_top1_accuracies, label='Train Top1 Accuracy', alpha=0.5)
    plt.plot(train_top5_accuracies, label='Train Top5 Accuracy', alpha=0.5)
    plt.plot(val_top1_accuracies, label='Val Top1 Accuracy', alpha=0.5)
    plt.plot(val_top5_accuracies, label='Val Top5 Accuracy', alpha=0.5)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    if args.tto:
        plt.title('Student Model Accuracies')
    elif not args.kd:
        plt.title('Without Knowledge Distillation Accuracies')
    elif args.kd:
        plt.title('Knowledge Distillation Accuracies')
    
    plt.grid()
    plt.legend(loc='best')
    
    if args.tto:
        plt.savefig(os.path.join(args.plots_path, 'Teacher Model Accuracy using {} {} {}.png'.format(args.teacher_type, args.dataset, args.num_classes)))
    elif not args.kd:
        plt.savefig(os.path.join(args.plots_path, 'Without Knowledge Distillation Accuracy using {} {}.png'.format(args.dataset, args.num_classes)))
    elif args.kd:
        plt.savefig(os.path.join(args.plots_path, 'Knowledge Distillation Accuracy using {} {} at temp {}.png'.format(args.dataset, args.num_classes, args.temp)))
    else:
        raise NotImplementedError