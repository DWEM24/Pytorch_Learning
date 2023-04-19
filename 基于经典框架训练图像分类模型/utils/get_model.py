from torchvision import models
from torch import nn


class Resnet18_ft:
    """Resnet18 fine-tuning"""
    def __init__(self):
        self.num_classes = 102
        self.feature_extract = True                                 # 是否只训练最后一层的参数
        self.use_pretrained = True                                  # 是否加载预训练模型的权重参数

    def initialize(self):
        model_ft = models.resnet18(pretrained=self.use_pretrained)  # pretrained=True 表示加载预训练模型的权重参数，自动下载到 /.cahe/torch/checkpoints

        if self.feature_extract:                                    # 冻住模型所有层的参数
            for param in model_ft.parameters():
                param.requires_grad = False

        num_ftrs = model_ft.fc.in_features                          # 获取最后一层的输入特征数
        model_ft.fc = nn.Linear(num_ftrs, self.num_classes)         # 重置最后一层的参数
        input_size = 64

        param_to_update = model_ft.parameters()                     # 把模型中所有层名字，权重保存下来
        print("Prameters to learn:")
        if self.feature_extract:
            param_to_update = []                                    # 如果只训练最后一层的参数，那么就不需要把所有层的参数都保存下来
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    param_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)
        # 输入图像的尺寸

        return model_ft, input_size, param_to_update