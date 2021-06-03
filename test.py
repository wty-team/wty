import time
import random
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from config import args
from models import StudentModel, TeacherModel
from utils import *


# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(args):

    # For Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare Data Loader #
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare Data #
    if args.num_classes == 10:
        test_cifar = CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)

    elif args.num_classes == 100:
        test_cifar = CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)

    test_loader = DataLoader(dataset=test_cifar, batch_size=args.batch_size, shuffle=False)

    # Model Configuration #
    if args.tto:
        model = TeacherModel(args.teacher_type, args.num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(args.weights_path, 'Best_Teacher_Model_{}_{}_{}.h5'.format(args.teacher_type, args.dataset, args.num_classes)), map_location='cuda:0'))
        print("Teacher Model is successfully loaded.")
    elif not args.kd:
        model = StudentModel(args.num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(args.weights_path, 'Best_Model_WithoutKD_{}_{}.h5'.format(args.dataset, args.num_classes)), map_location='cuda:0'))
        print("Model without Knowledge Distillation is successfully loaded.")
    elif args.kd:
        model = StudentModel(args.num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(args.weights_path, 'Best_Model_KD_{}_{}.h5'.format(args.dataset, args.num_classes)), map_location='cuda:0'))
        print("Model with Knowledge Distillation is successfully loaded.")
        #model = torch.load(model_path, map_location='cuda:0')

    # Average Meter #
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Switch to Evaluation Mode #
    model.eval()

    # For Time Measurement #
    t1 = time.time()

    for i, (image, label) in enumerate(test_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # For Inference #
        with torch.no_grad():
            pred = model(image)
        
        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

    # For Time Measurement #
    t2 = time.time()

    # Print Statistics #
    print("Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}% | Time Taken {:.2f}s".format(top1_accuracy.avg, top5_accuracy.avg, t2-t1))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test(args)