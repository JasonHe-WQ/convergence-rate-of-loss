from torchsummary import summary
import modifiedResnet18
import utils


# 加载模型
model = modifiedResnet18.resnet18_cbam()
# 使用CIFAR10数据集的输入形状来打印模型摘要
summary(model, input_size=(3, 32, 32))

# 请注意，这里的输入形状是(3, 32, 32)，而不是(3, 224, 224)。这是因为CIFAR10数据集的图像尺寸为32x32，而不是ImageNet的224x224。

baseModel = utils.initialize_model()
summary(baseModel, input_size=(3, 32, 32))

