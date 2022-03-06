
import paddle
from paddle.nn import Layer
from paddle.nn import Conv2D, BatchNorm2D,AdaptiveAvgPool2D,Sequential,Upsample,ReLU,Sequential

import os, sys
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

from backbone.ResNet import ResNet50

class Conv_bn(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_bn, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm2D(num_features=out_channels)
        self.relu = ReLU()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PSPModule(Layer):
    def __init__(self, in_channels, bin_size, out_shape):
        super(PSPModule, self).__init__()
        length = len(bin_size)
        out_channels = int(in_channels/length)
        self.bins = []
        for i in range(length):
            bin_layer = Sequential(
                AdaptiveAvgPool2D(output_size=(bin_size[i],bin_size[i])),
                Conv_bn(in_channels = in_channels, out_channels= out_channels, kernel_size=1),
                Upsample(size=out_shape, mode='BILINEAR', align_corners=True)
            )
            self.bins.append(bin_layer)

    def forward(self, inputs):
        bin_out = [inputs]
        for bin_layer in self.bins:
            bin_out.append(bin_layer(inputs))
        return paddle.concat(bin_out, axis=1)


class PSPNet(Layer):
    def __init__(self, num_class):
        super(PSPNet, self).__init__()
        resnet = ResNet50()
        self.conv1 = resnet.conv1
        self.pool1 = resnet.pool1
        self.conv2 = resnet.conv2
        self.conv3 = resnet.conv3
        self.conv4 = resnet.conv4
        self.conv5 = resnet.conv5
        self.pspmodule = PSPModule(2048, bin_size=[1,2,3,6], out_shape=[64,64])
        self.class_conv = Sequential(
            Conv_bn(in_channels=4096, out_channels=512, kernel_size=3, padding=1),
            Conv_bn(in_channels=512, out_channels=num_class, kernel_size=1)
        )
        self.up = Upsample(size=(512, 512), mode='BILINEAR', align_corners=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pspmodule(x)
        x = self.class_conv(x)
        x = self.up(x)

        return x


def main():
    import numpy as np
    data = np.random.randn(1, 3, 512, 512)
    data = paddle.to_tensor(data).astype('float32')
    model = PSPNet(19)
    model.train()
    out = model(data)
    print(out.shape)

if __name__ == '__main__':
    main()


