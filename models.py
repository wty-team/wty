import torch
import torch.nn as nn
from torchvision.models import vgg19_bn,vgg16, resnet34,resnet50


class StudentModel(nn.Module):
    """Student Model"""
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()

        self.num_classes = num_classes

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*4*4, out_features=num_classes)
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def TeacherModel(sort, num_classes):
    if sort == 'vgg19':
        model = vgg19_bn(pretrained=False)
        model.classifier = nn.Linear(in_features=model.classifier[0].in_features, out_features=num_classes)
    elif sort == 'vgg16':
        model = vgg16(pretrained=False)
        model.classifier = nn.Linear(in_features=model.classifier[0].in_features, out_features=num_classes)
    elif sort == 'resnet34':
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif sort == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model