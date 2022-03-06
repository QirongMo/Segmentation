
import paddle
from paddle.nn import Conv2D, BatchNorm2D, ReLU, Sequential, Dropout, Upsample, AdaptiveAvgPool2D
from paddle.nn import Layer


import os, sys
o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径
from backbone.Xception import Xception

class Conv_bn(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 act=True):
        super(Conv_bn, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, dilation=dilation, padding=padding, groups=groups)

        self.bn = BatchNorm2D(num_features=out_channels)
        if act:
            self.relu = ReLU()
        else:
            self.relu = None

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ASPP(Layer):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.conv1 = Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.dila_conv = []
        for i in range(len(rates)):
            self.dila_conv.append(
                Sequential(

                    Conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                            padding=rates[i], dilation=rates[i], groups=in_channels),
                    Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                )
            )

        self.adapt_pool = AdaptiveAvgPool2D(output_size=(1, 1))
        self.pool_conv = Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.project = Conv_bn(in_channels=out_channels * (2 + len(rates)), out_channels=out_channels, kernel_size=1)

    def forward(self, inputs):
        _, _, h, w = inputs.shape
        feature = []
        feature.append(self.conv1(inputs))

        for dila_conv in self.dila_conv:
            feature.append(dila_conv(inputs))

        adapt_pool = self.adapt_pool(inputs)
        pool_conv = self.pool_conv(adapt_pool)
        feature.append(Upsample((h, w), mode='BILINEAR')(pool_conv))

        x = paddle.concat(feature, axis=1)
        x = self.project(x)
        return x


class DeepLabV3Plus(Layer):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        backbone = Xception()
        # entry flow
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # midflow
        self.layer5 = backbone.layer5

        # exitflow
        self.layer6 = backbone.exitflow

        self.classifier = Sequential(
            Conv_bn(in_channels=432, out_channels=256, kernel_size=3, padding=1),
            Conv_bn(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            Conv_bn(in_channels=256, out_channels=num_classes, kernel_size=1),
        )
        self.aspp = ASPP(in_channels=2048, out_channels=256, rates=[6, 12, 18])
        self.lowlevel2 = Conv_bn(in_channels=2048, out_channels=48, kernel_size=1)

    def forward(self, inputs):
        # Entry Flow
        x = self.layer1(inputs)  # 1/2
        x = self.layer2(x)  # 1/4
        lowlevel1 = x
        x = self.layer3(x)  # 1/8
        x = self.layer4(x)  # 1/16
        x = self.layer5(x)  # 1/16
        x = self.layer6(x)  # 1/16
        lowlevel2 = x
        # ASPP
        x = self.aspp(x)
        x = Upsample((lowlevel2.shape[2], lowlevel2.shape[3]), mode='BILINEAR')(x)
        lowlevel2 = self.lowlevel2(lowlevel2)
        x = paddle.concat([lowlevel2, x], axis=1)
        x = Upsample((lowlevel1.shape[2], lowlevel1.shape[3]), mode='BILINEAR')(x)

        x = paddle.concat([x, lowlevel1], axis=1)
        x = self.classifier(x)
        x = Upsample((inputs.shape[2], inputs.shape[3]), mode='BILINEAR')(x)
        return x


def main():
    data = paddle.rand((1, 3, 256, 256))
    model = DeepLabV3Plus(19)
    pred = model(data)
    print(pred.shape)


if __name__ == '__main__':
    main()