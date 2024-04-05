import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


# 1. 数据加载函数
def load_data(batch_size, dtype=torch.float32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader



# 2. 模型初始化函数
def initialize_model(dtype=torch.float32):
    resnet18 = models.resnet18(weights=None)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 10)
    return resnet18.to(dtype=dtype).cuda()


# 3. 训练函数
def train_model(model, trainloader, criterion, optimizer, num_epochs=10, dtype=torch.float32):
    model.train()
    loss_history = []
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda().to(dtype=dtype), labels.cuda()
            if i == 0:
                print(f"getting started with precision f{inputs[0].dtype}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss.item()}')
        scheduler.step()
    return loss_history
