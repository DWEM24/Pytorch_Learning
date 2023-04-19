import time
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import copy

from .dataloader import ImageData
from .get_model import Resnet18_ft


class TrainModel:
    def __init__(self, num_epochs=25, batch_size=64):

        self.model, self.input_size, self.param_to_update = Resnet18_ft().initialize()
        self.image_datasets, self.dataloaders, self.dataset_sizes, self.class_names = ImageData(batch_size).transform()  # 调用类一定是要先创建类的实例
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.param_to_update, lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.num_epochs = num_epochs
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_to_save = "model.pth"

    def train(self):
        start = time.time()
        print('Training on {}'.format(self.device))

        train_acc_history = []  # 记录每次迭代的训练集准确率
        val_acc_history = []  # 记录每次迭代的验证集准确率
        train_losses = []
        val_losses = []

        best_acc = 0.0                                           # 目前最高的验证集准确率

        LRs = [(self.optimizer.param_groups[0]['lr'])]           # 学习率

        best_model_wts = copy.deepcopy(self.model.state_dict())  # 定一个一个变量保存模型的权重，开始的时候先初始化为当前模型的权重

        for epoch in range(self.num_epochs):
            print("[Epoch {}/{}]".format(epoch+1, self.num_epochs))
            print('\033[34m——\033[0m'*50)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # 训练模式
                    print('\033[34mTraining...\033[0m')
                else:
                    self.model.eval()  # 验证模式
                    print('\033[34mValidating...\033[0m')

                running_loss = 0.0
                running_corrects = 0

                #  迭代数据, loader中已经确定了batch_size
                for inputs, labels in tqdm(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # print(_, preds)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)    # loss.item()是一个数，inputs.size(0)是一个batch_size
                    running_corrects += torch.sum(preds == labels.data)

                # 计算损失和准确率
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                # 打印、保存每一次迭代的损失和准确率
                time_elapsed = time.time() - start  # 一个epoch的时间
                print('\033[34m{} Loss:\033[0m {:.4f}||\033[34mAcc:\033[0m {:.4f}||\033[34mTime:\033[0m {:.0f}m {:.0f}s'.format(
                    phase, epoch_loss, epoch_acc, time_elapsed // 60, time_elapsed % 60
                ))
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)
                    val_losses.append(epoch_loss)

                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                    train_losses.append(epoch_loss)

                # 记录验证集训练效果最好的那个模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    state = {
                        'state_dict': self.model.state_dict(),  # 字典的key是层名字，value是层的权重
                        'bst_acc': best_acc,  # 记录最佳准确率
                        'optimizer': self.optimizer.state_dict(),  # 优化器的状态
                    }
                    torch.save(state, self.path_to_save)

            print('optimizer learning rate: {}'.format(self.optimizer.param_groups[0]['lr']))
            LRs.append(self.optimizer.param_groups[0]['lr'])
            print()
            self.scheduler.step()  # 更新学习率， 累计到制定好的step_size，就会更新学习率，这里第8个epoch会更新

        # 整个模型的训练时间、最佳准确率
        time_elapsed = time.time() - start
        print('\033[31mTraining complete in\033[0m {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
        print('\033[31mBest val Acc:\033[0m {:4f}'.format(best_acc))

        # 加载最好的模型，返回
        model = self.model.load_state_dict(best_model_wts)
        return model, train_acc_history, val_acc_history, train_losses, val_losses, LRs