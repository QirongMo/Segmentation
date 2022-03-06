


import paddle
from paddle.nn import Layer
from paddle.nn import AdaptiveAvgPool2D, Sequential, Upsample

import os, sys
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径

from backbone.ResNet_Multi_Gride import ResNet50, Conv_bn

class ASPPplus(Layer):
    def __init__(self, in_channels, out_channels, dila):
        super(ASPPplus, self).__init__()
        self.conv1 = Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        length = len(dila)
        self.dila_conv = []
        for i in range(length):
            self.dila_conv.append(
                Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=dila[i], padding=dila[i])
            )

        self.pool = AdaptiveAvgPool2D((1,1))
        self.conv2 = Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.up = Upsample((64,64))

    def forward(self, inputs):
        out = [self.conv1(inputs)]
        for dila_conv in self.dila_conv:
            out.append(dila_conv(inputs))
        out.append(self.up(self.conv2(self.pool(inputs))))
        return paddle.concat(out, axis=1)


class DeepLabV3(Layer):
    def __init__(self, num_class):
        super(DeepLabV3, self).__init__()
        resnet = ResNet50()
        self.conv1 = resnet.conv1
        self.pool1 = resnet.pool1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer5 = resnet.layer5
        self.layer6 = resnet.layer6
        self.layer7 = resnet.layer7
        self.asppplus = ASPPplus(in_channels=2048, out_channels=256, dila=[6,12,18])
        self.class_conv = Sequential(
            Conv_bn(in_channels=1280, out_channels=256, kernel_size=1),
            Conv_bn(in_channels=256, out_channels=num_class, kernel_size=1, act=False)
        )
        self.up = Upsample(size=(512, 512), mode='BILINEAR', align_corners=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.asppplus(x)
        x = self.class_conv(x)
        x = self.up(x)

        return x


def main():
    data = paddle.rand((1, 3, 512, 512))
    model = DeepLabV3(19)
    model.eval()
    out = model(data)
    print(out.shape)

if __name__ == '__main__':
    main()


