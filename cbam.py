import torch
import utils
import modifiedResnet18
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

def main(models):
    lossList = []
    dtype = torch.float32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    batch_size = 64
    for model_name in models:
        print('-' * 20, )
        print(f'Batch Size: {batch_size}')
        trainloader = utils.load_data(batch_size, dtype)
        if model_name == "base":
            model = utils.initialize_model()
        else:
            model = modifiedResnet18.resnet18_cbam()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_history = utils.train_model(model, trainloader, criterion, optimizer, 15, dtype)
        # 保存loss数据
        lossList.append(loss_history)
        plt.plot(loss_history, label=f'Batch Size: {batch_size}')
        print('\n' * 2)

    # 存储数据为csv文件
    import pandas as pd
    df = pd.DataFrame(lossList, index=models)
    df.to_csv('loss_cbam.csv')

    # 绘制图像
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration for Different Models')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # 执行代码
    models = ["cbam","base",]  # 调整这个列表来探索不同的批次大小
    main(models)