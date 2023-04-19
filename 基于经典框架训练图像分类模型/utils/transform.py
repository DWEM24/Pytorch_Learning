from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import os
import torch


class ImageData(object):
    """获取模型的训练输入"""
    def __init__(self, batch_size=64, data_dir='../data/102flowers/', ):
        self.data_dir = data_dir
        self.bactch_size = batch_size
        self.data = ['train', 'valid', 'test']
        self.data_transforms = {
            'train':
                transforms.Compose(
                    [
                        transforms.Resize([96, 96]),  # 缩放图片为96*96的大小
                        transforms.RandomRotation(45),  # 随机旋转45度(-45,45)之间随机选
                        transforms.CenterCrop(64),  # 从中心裁剪64*64的图片，64是组中图片的大小
                        transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转
                        transforms.RandomVerticalFlip(0.5),  # 随机垂直翻转
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        # 随机改变亮度、对比度、饱和度和色调
                        transforms.RandomGrayscale(p=0.025),  # 随机将图片转为灰度图
                        transforms.ToTensor(),  # 将图片转为Tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
                    ]
                ),
            'valid':
                transforms.Compose(
                    [
                        transforms.Resize([64, 64]),  # 缩放图片为96*96的大小
                        transforms.ToTensor(),  # 将图片转为Tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        # 标准化要和训练集的一样
                        # 验证集合是否需要做数据增强，取决于训练好模型之后实际的使用场景中是什么情况。
                        # 例如。如果预测的环境曝光值比较高，就需要我们对验证集做一些数据增强，比如亮度、对比度、饱和度等
                        # 再比如，如果预测的环境中有很多噪声，就需要我们对验证集做一些数据增强，比如高斯噪声、椒盐噪声等
                        # 再比如，如果预测的环境中只能看到一部分图像，就需要我们对验证集做一些数据增强，比如裁剪、旋转等
                    ]
                ),
            'test':
                transforms.Compose(
                    [
                        transforms.Resize([64, 64]),  # 缩放图片为96*96的大小
                        transforms.ToTensor(),  # 将图片转为Tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        # 标准化要和训练集的一样
                        # 验证集合是否需要做数据增强，取决于训练好模型之后实际的使用场景中是什么情况。
                        # 例如。如果预测的环境曝光值比较高，就需要我们对验证集做一些数据增强，比如亮度、对比度、饱和度等
                        # 再比如，如果预测的环境中有很多噪声，就需要我们对验证集做一些数据增强，比如高斯噪声、椒盐噪声等
                        # 再比如，如果预测的环境中只能看到一部分图像，就需要我们对验证集做一些数据增强，比如裁剪、旋转等
                    ]
                )
        }

    def transform(self, ):
        image_datasets = {x: ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in self.data}
        dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.bactch_size, shuffle=True, num_workers=0) for x in self.data}
        dataset_sizes = {x: len(image_datasets[x]) for x in self.data}
        class_names = image_datasets['train'].classes
        return image_datasets, dataloader, dataset_sizes, class_names
