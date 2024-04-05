import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import utils


# 主执行函数
def main(dtypes, batch_size=64, num_epochs=10):
    lossList = []
    # 打开TF32

    for dtype in dtypes:
        print(f'\n\n=== Training with {dtype} ===')
        trainloader = utils.load_data(batch_size, dtype)
        model = utils.initialize_model(dtype)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_history = utils.train_model(model, trainloader, criterion, optimizer, num_epochs, dtype)

        # 绘制图像
        plt.plot(loss_history, label=f'Data Type: {dtype}')
        lossList.append(loss_history)

    # 存储为csv文件
    import pandas as pd
    df = pd.DataFrame(lossList, index=dtypes)
    df.to_csv('loss_precision.csv')

    # 绘制图像
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Iteration with Different Data Types (Batch Size: {batch_size})')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dtypes = [torch.bfloat16, torch.float32, torch.float64, ]  # FP64, TF32, BF16
    main(dtypes)
