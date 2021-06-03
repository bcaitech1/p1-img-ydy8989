import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers.classifier import ClassifierHead
from torch import nn
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class CustomResNext(nn.Module):
    def __init__(self, model_arch, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.num_features
        self.mask_classifier = ClassifierHead(n_features, 3)
        self.gender_classifier = ClassifierHead(n_features, 2)
        self.age_classifier = ClassifierHead(n_features, 3)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model.forward_features(x)
        z = self.age_classifier(x)
        y = self.gender_classifier(x)
        x = self.mask_classifier(x)
        return x, y, z




class EffiNetb0(nn.Module):
    def __init__(self, num_classes):
        super(EffiNetb0, self).__init__()
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, num_classes), # 1280 b0, 1536 b3, 1792 b4
        )

    def forward(self, x):
        return self.backbone(x)

class EffiNetb3(nn.Module):
    def __init__(self, num_classes):
        super(EffiNetb3, self).__init__()
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = EfficientNet.from_name('efficientnet-b3')
        self.backbone._fc = nn.Sequential(
            nn.Linear(1536, num_classes), # 1280 b0, 1536 b3, 1792 b4
        )

    def forward(self, x):
        return self.backbone(x)

class EffiNetb4(nn.Module):
    def __init__(self, num_classes):
        super(EffiNetb4, self).__init__()
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = EfficientNet.from_name('efficientnet-b4')
        self.backbone._fc = nn.Sequential(
            nn.Linear(1792, num_classes), # 1280 b0, 1536 b3, 1792 b4
        )
    def forward(self, x):
        return self.backbone(x)


class EffiNetb6(nn.Module):
    def __init__(self, num_classes):
        super(EffiNetb6, self).__init__()
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = EfficientNet.from_name('efficientnet-b6')
        self.backbone._fc = nn.Sequential(
            nn.Linear(2304, num_classes), # 1280 b0, 1536 b3, 1792 b4
        )

    def forward(self, x):
        return self.backbone(x)

class NfNet(nn.Module):
    def __init__(self, num_classes):
        super(NfNet, self).__init__()
        self.backbone = timm.create_model('nfnet_l0c',
                                          pretrained=True)  # EfficientNet.from_pretrained('efficientnet-b3')

        self.backbone.head.fc = nn.Sequential(
            nn.Linear(2304, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)