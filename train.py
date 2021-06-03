import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from config import args
from models import TeacherModel, StudentModel
from utils import *


# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):

    # For Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Weights and Plots Path #
    paths = [args.data_path, args.weights_path, args.plots_path]
    for path in paths:
        make_dirs(path)
        
    # Define Transform for Image Processing #
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare Data #
    if args.num_classes == 10:
        train_cifar = CIFAR10(root=args.data_path, train=True, transform=train_transform, download=True)
        val_cifar = CIFAR10(root=args.data_path, train=False, transform=val_transform, download=True)

    elif args.num_classes == 100:
        train_cifar = CIFAR100(root=args.data_path, train=True, transform=train_transform, download=True)
        val_cifar = CIFAR100(root=args.data_path, train=False, transform=val_transform, download=True)

    train_loader = DataLoader(dataset=train_cifar, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_cifar, batch_size=args.batch_size, shuffle=False)

    # Model Specification #
    if args.tto:
        model = TeacherModel(args.teacher_type, args.num_classes).to(device)
    elif not args.kd:
        model = StudentModel(args.num_classes).to(device)
    elif args.kd:
        teacher_model = TeacherModel(args.teacher_type, args.num_classes).to(device)
        teacher_model.load_state_dict(torch.load(os.path.join(args.weights_path, 'Best_Teacher_Model_{}_{}_{}.h5'.format(args.teacher_type, args.dataset, args.num_classes))))
        model = StudentModel(args.num_classes).to(device)

    # Weight Initialization #
    if args.init == 'normal':
        model.apply(init_weights_normal)
    elif args.init == 'xavier':
        model.apply(init_weights_xavier)
    elif args.init == 'he':
        model.apply(init_weights_kaiming)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = nn.CrossEntropyLoss()

    # Optimizer #
    if args.num_classes == 10:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, args)

    elif args.num_classes == 100:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer_scheduler = get_lr_scheduler('step', optimizer, args)

    # Constants #
    best_top1_accuracy = 0

    # Lists #
    train_losses, train_top1_accuracies, train_top5_accuracies = list(), list(), list()
    val_losses, val_top1_accuracies, val_top5_accuracies = list(), list(), list()

    # Train and Validation Start #
    if args.tto:
        print("Training only Teacher Model of {} has started with total epoch of {}.".format(args.teacher_type, args.num_epochs))
    elif not args.kd:
        print("Training without Knowledge Distillation has started with total epoch of {}.".format(args.num_epochs))
    elif args.kd:
        print("Training with Knowledge Distillation has started with total epoch of {}.".format(args.num_epochs))
    else:
        raise NotImplementedError

    # For Time Measurement #
    t1 = time.time()

    for epoch in range(args.num_epochs):

        # Train #
        if args.tto:
            train_loss, train_top1_accuracy, train_top5_accuracy = train(train_loader, model, criterion, optimizer, epoch, args)
        elif not args.kd:
            train_loss, train_top1_accuracy, train_top5_accuracy = train(train_loader, model, criterion, optimizer, epoch, args)
        elif args.kd:
            train_loss, train_top1_accuracy, train_top5_accuracy = train_with_teacher(train_loader, teacher_model, model, optimizer, epoch, args)
        
        train_losses.append(train_loss)
        train_top1_accuracies.append(train_top1_accuracy)
        train_top5_accuracies.append(train_top5_accuracy)

        optimizer_scheduler.step()

        # Validation #
        val_loss, val_top1_accuracy, val_top5_accuracy = validate(val_loader, model, criterion, epoch, args)
        
        val_losses.append(val_loss)
        val_top1_accuracies.append(val_top1_accuracy)
        val_top5_accuracies.append(val_top5_accuracy)

        # Best Top 1 Accuracy #
        if val_top1_accuracy > best_top1_accuracy:
            print("The best model has been updated!")
            best_top1_accuracy = max(val_top1_accuracy, best_top1_accuracy)

            # Save Models #
            if args.tto:
                torch.save(model.state_dict(), os.path.join(args.weights_path, 'Best_Teacher_Model_{}_{}_{}.h5'.format(args.teacher_type, args.dataset, args.num_classes)))
            elif not args.kd:
                torch.save(model.state_dict(), os.path.join(args.weights_path, 'Best_Model_WithoutKD_{}_{}.h5'.format(args.dataset, args.num_classes)))
            elif args.kd:
                torch.save(model.state_dict(), os.path.join(args.weights_path, 'Best_Model_KD_{}_{}.h5'.format(args.dataset, args.num_classes)))

        print("Best Top 1 Accuracy {:.2f}% So far.\n".format(best_top1_accuracy))

    # Plot Losses and Accuracies #
    plot_losses(train_losses, val_losses, args)
    plot_accuracies(train_top1_accuracies, train_top5_accuracies, val_top1_accuracies, val_top5_accuracies, args)

    # For Time Measurement #
    t2 = time.time()

    print("The whole training process took {:.2f}s".format(t2-t1))


def train(data_loader, model, criterion, optimizer, epoch, args):

    # For Time Measurement #
    t1 = time.time()

    # Average Meter #
    top_loss = AverageMeter()
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Total Batch Size #
    total_batch = len(data_loader)

    # Switch to Train Mode #
    model.train()

    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # Initialize Optimizer #
        optimizer.zero_grad()

        # Forward Data and Calculate Predictions #
        pred = model(image)
        loss = criterion(pred, label)

        # Back Propagation and Update #
        loss.backward()
        optimizer.step()

        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
        top_loss.update(loss.item(), image.size(0))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

        # For Time Measurement #
        t2 = time.time()

        # Print Statistics #
        if (i+1) % args.print_every == 0:
            print("Train | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.2f} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}% | Time Taken {:.2f}s"
                  .format(epoch+1, args.num_epochs, i+1, total_batch, top_loss.avg, top1_accuracy.avg, top5_accuracy.avg, t2-t1))

    return top_loss.avg, top1_accuracy.avg, top5_accuracy.avg


def validate(data_loader, model, criterion, epoch, args):

    # For Time Measurement #
    t1 = time.time()

    # Average Meter #
    top_loss = AverageMeter()
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Total Batch Size #
    total_batch = len(data_loader)

    # Switch to Evaluation Mode #
    model.eval()

    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # For Inference #
        with torch.no_grad():
            pred = model(image)
            loss = criterion(pred, label)

        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
        top_loss.update(loss.item(), image.size(0))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

        # For Time Measurement #
        t2 = time.time()

        # Print Statistics #
        if (i+1) % args.print_every == 0:
            print("Val | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.2f} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}% | Time Taken {:.2f}s\n"
                  .format(epoch+1, args.num_epochs, i+1, total_batch, top_loss.avg, top1_accuracy.avg, top5_accuracy.avg, t2-t1))

    return top_loss.avg, top1_accuracy.avg, top5_accuracy.avg


def train_with_teacher(data_loader, teacher_model, student_model, optimizer, epoch, args):

    # For Time Measurement #
    t1 = time.time()

    # Average Meter #
    top_loss = AverageMeter()
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Total Batch Size #
    total_batch = len(data_loader)

    # Switch to Train Mode #
    teacher_model.train()
    student_model.train()

    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # Initialize Optimizer #
        optimizer.zero_grad()

        # Forward Data and Calculate Predictions #
        pred_teacher = teacher_model(image)
        pred_student = student_model(image)
        loss = kd_loss(pred_student, label, pred_teacher, alpha=args.alpha, temp=args.temp)

        # Back Propagation and Update #
        loss.backward()
        optimizer.step()

        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred_student.data, label.data, topk=(1, 5))
        top_loss.update(loss.item(), image.size(0))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

        # For Time Measurement #
        t2 = time.time()

        # Print Statistics #
        if (i+1) % args.print_every == 0:
            print("Train-KD | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.2f} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}% | Time Taken {:.2f}s"
                  .format(epoch+1, args.num_epochs, i+1, total_batch, top_loss.avg, top1_accuracy.avg, top5_accuracy.avg, t2-t1))

    return top_loss.avg, top1_accuracy.avg, top5_accuracy.avg


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main(args)