import argparse
    
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')

parser.add_argument('--teacher_type', type=str, default='resnet50', help='which model to use for teacher model', choices=['vgg16','resnet50','vgg19', 'resnet34'])
parser.add_argument('--dataset', type=str, default='CIFAR', help='which dataset to train')
parser.add_argument('--num_classes', type=int, default=100, help='num_classes for cifar dataset', choices=[10, 100])
parser.add_argument('--init', type=str, default='he', help='which initialization technique to apply', choices=['normal', 'xavier', 'he'])

parser.add_argument('--data_path', type=str, default='./data/', help='data path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')

parser.add_argument('--alpha', default=0.2, type=float, help='alpha for Kullback Leibler Divergence Loss')
parser.add_argument('--temp', default=20, type=int, help='temperature for Kullback Leibler Divergence Loss')
parser.add_argument('--tto', default=False, type=bool, help='Train teacher model only')
parser.add_argument('--kd', default=True, type=bool, help='Activate knowledge distillation')

parser.add_argument('--num_epochs', type=int, default=150, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every n iteration')
parser.add_argument('--save_every', type=int, default=5, help='save model weights for every n epoch')

parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

args = parser.parse_args()


if __name__ == "__main__":
    print(args)