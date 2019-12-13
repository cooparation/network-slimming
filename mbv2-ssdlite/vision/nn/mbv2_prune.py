import torch.nn as nn
import math
from collections import OrderedDict

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)),
            ('conv/bn', nn.BatchNorm2d(oup)),
            ('conv/relu', ReLU(inplace=True))
            ])
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(OrderedDict([
            ('Conv_1', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
            ('Conv_1/bn', nn.BatchNorm2d(oup)),
            ('Conv_1/relu', ReLU(inplace=True))
            ])
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )

class InvertedResidual_Prune(nn.Module):
    #def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
    def __init__(self, inp, hidden_dim, oup, stride, expand_ratio, layer_id, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual_Prune, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        #hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(OrderedDict([
                    # dw
                    ('conv/dw', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                    ('conv/dw/bn', nn.BatchNorm2d(hidden_dim)),
                    ('conv/dw/relu', ReLU(inplace=True)),
                    # pw-linear
                    ('conv/linear', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                    ('conv/linear/bn', nn.BatchNorm2d(oup)),
                    ])
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            conv_name = 'conv_'+str(layer_id)
            if use_batch_norm:
                self.conv = nn.Sequential(OrderedDict([
                    # pw
                    (conv_name+'/pw',nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                    (conv_name + '/pw/bn', nn.BatchNorm2d(hidden_dim)),
                    (conv_name + '/pw/relu', ReLU(inplace=True)),
                    # dw
                    (conv_name + '/dw', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                    (conv_name + '/dw/bn', nn.BatchNorm2d(hidden_dim)),
                    (conv_name +'/dw/relu', ReLU(inplace=True)),
                    # pw-linear
                    (conv_name+'/linear', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                    (conv_name+'/linear/bn', nn.BatchNorm2d(oup)),
                    ])
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Prune(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False, cfg=None,
                 in_channel=32, last_channel=1280):
        super(MobileNetV2_Prune, self).__init__()
        block = InvertedResidual_Prune
        input_channel = in_channel #24 #32
        #last_channel = last_channel #713 #1280
        if cfg == None:
            input_channel = 32
            last_channel = 1280
            interverted_residual_setting = [
                # hidden_c, out_c, s, expand_ratio
                [input_channel, 16, 1, 1],
                [96,  24,  2, -1],
                [144, 24,  2, -1],
                [144, 32,  2, -1],
                [192, 32,  2, -1],
                [192, 32,  2, -1],
                [192, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 64,  2, -1],
                [384, 96,  1, -1],
                [576, 96,  1, -1],
                [576, 96,  1, -1],
                [576, 160, 2, -1], # conv_13/expand
                [960, 160, 2, -1],
                [960, 160, 2, -1],
                [960, 320, 1, -1],
            ]
        else:
            interverted_residual_setting = cfg

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        layer_id = 0
        for hidden_c, out_c, s, t in interverted_residual_setting:
            output_channel = int(out_c * width_mult)
            #for i in range(n):
            if layer_id in [0, 1, 3, 6, 10, 13, 16]:
                self.features.append(block(input_channel, hidden_c, output_channel, s,
                                           t, layer_id, use_batch_norm=use_batch_norm,
                                           onnx_compatible=onnx_compatible))
            else:
                self.features.append(block(input_channel, hidden_c, output_channel, 1,
                                           t, layer_id, use_batch_norm=use_batch_norm,
                                           onnx_compatible=onnx_compatible))
            input_channel = output_channel
            layer_id += 1
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
