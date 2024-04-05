import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import utils


# 4. 主执行函数
def main(batch_sizes):
    lossList = []
    dtype = torch.float32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for batch_size in batch_sizes:
        print('-' * 20, )
        print(f'Batch Size: {batch_size}')
        trainloader = utils.load_data(batch_size, dtype)
        model = utils.initialize_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_history = utils.train_model(model, trainloader, criterion, optimizer, 15, dtype)
        # 保存loss数据
        lossList.append(loss_history)
        plt.plot(loss_history, label=f'Batch Size: {batch_size}')
        print('\n' * 2)

    # 存储数据为csv文件
    import pandas as pd
    df = pd.DataFrame(lossList, index=batch_sizes)
    df.to_csv('loss_batch.csv')

    # 绘制图像
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration for Different Batch Sizes')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # 执行代码
    batch_sizes = [64, 32, 16]  # 调整这个列表来探索不同的批次大小
    main(batch_sizes)
