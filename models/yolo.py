# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules
这个模块是yolov5的模型搭建模块，非常的重要，不过代码量并不大，不是很难，只是yolov5的作者把封装的太好了，
模型扩展了很多的额外的功能，导致看起来很难，其实真正有用的代码不多的。重点是抓住三个函数是在哪里调用的，谁调用谁的，理解这个应该就不会很难。
Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""
from models.detect_t import TDetect

import argparse   # 解析命令行参数模块
import sys   # sys系统模块 包含了与Python解释器和它的环境有关的函数
from copy import deepcopy   # 数据拷贝模块 深拷贝
from pathlib import Path   # Path将str转换为Path对象 使字符串路径易于操作的模块

FILE = Path(__file__).resolve() # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\modles\yolo.py'
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import * #把写好的模块全部导入
from models.experimental import * #导入实验模块
from models.cspcm import *

#都是utils包的内容
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)

# 导入thop包 用于计算FLOPs（FLOPS用于衡量硬件性能指标）
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

#Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape，为后面的计算损失或者NMS作准备。
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
               detection layer 相当于yolov3中的YOLOLayer层
               :params nc: number of classes
               :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
               :params ch: [128, 256, 512] 3个输出feature map的channel
        """
        super().__init__()
        self.nc = nc  # number of classes VOC: 20
        self.no = nc + 5  # number of outputs per anchor  VOC: 5+20=25  xywhc+20classes 每个anchor的输出
        self.nl = len(anchors)  # number of detection layers Detect的个数 3,我们的模型是4个
        self.na = len(anchors[0]) // 2  # number of anchors 每个feature map的anchor个数 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid {list: 3}  tensor([0.]) X 3
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # anchors=[3,6]  a=[3, 3, 2]  anchors以[w, h]对的形式存储  3个feature map 每个feature map上有三个anchor（w,h）
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # register_buffer
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter; 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        # output conv 对每个输出的feature map都要调用一次conv1x1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 将输出通过卷积到 self.no * self.na 的通道，达到全连接的作用
        # use in-place ops (e.g. slice assignment) 一般都是True 默认不使用AWS Inferentia加速
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
                :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                               分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                """
        z = []  # inference output
        for i in range(self.nl):# 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])  # conv  xi[bs, 128/256/512, 80, 80] to [bs, 75, 80, 80]？？？
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # [bs, 75, 80, 80] to [1, 4, 25, 80, 80] to [1, 4, 80, 80, 25]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了记录每个grid的网格坐标 方面后面使用
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    # 默认执行 不使用AWS Inferentia
                    # 这里的公式和yolov3、v4中使用的不一样 是yolov5作者自己用的 效果更好
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z是一个tensor list 三个元素 分别是[1, 19200, 25] [1, 4800, 25] [1, 1200, 25]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    #构造网格
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
                :params cfg:模型配置文件
                :params ch: input img channels 一般是3 RGB文件
                :params nc: number of classes 数据集的类别个数
                :anchors: 一般是None
        """
        # 加载配置文件
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict 传入的是字符串 因为没有import yaml
        else:  # is *.yaml 一般执行这里
            import yaml  # for torch hub 用于加载yaml后缀的文件
            self.yaml_file = Path(cfg).name   # cfg file name = yolov5s.yaml
            with open(cfg, errors='ignore') as f:
                # model dict  取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model# input channels  ch=3
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 3  先从yaml中去找，如果没有ch,则默认使用传过来的ch，并向yaml中写入ch
        # 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist   调用parse_model模块，传入yaml参数，将通道数用列表的形式传入
        # default class names ['0', '1', '2',..., '19']
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default class names ['0', '1', '2',..., '19']
        # self.inplace=True  默认True  不使用加速推理
        # AWS Inferentia Inplace compatiability
        # https://github.com/ultralytics/yolov5/pull/2953
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride 256是为了计算放缩另外设置的一个值，也可以设置为128或者别的8的倍数
            m.inplace = self.inplace
            # 计算三个feature map下采样的倍率  [8, 16, 32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward根据输入尺寸和预测尺寸推断步长
            # 求出相对当前feature map的anchor大小 如[10, 13]/8 -> [1.25, 1.625]
            m.anchors /= m.stride.view(-1, 1, 1)# anchor是像素坐标，但是使用的时候必须除以相同的步长
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once初始化偏置

            # 以下为新增代码 ------------- 以上为原有代码🚀
        elif isinstance(m, (TDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (TDetect)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            if isinstance(m, TDetect):
                m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)# 调用torch_utils.py下initialize_weights初始化模型权重
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # augmented inference, None  上下flip/左右flip
        # 是否在测试时也使用数据增强  Test Time Augmentation(TTA)
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train# 默认执行 正常前向推理

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 1, 0.83, 0.83, 0.67, 0.67]  # scales ratio
        f = [None, 3, None, 3, None, 3]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # _descale_pred将推理结果恢复到相对原图图片尺寸
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
                :params x: 输入图像
                :params profile: True 可以做一些性能评估
                :params feature_vis: True 可以做一些特征可视化
                :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                               分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                        inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                """
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        for m in self.model:
            # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
            # if not from previous layer   m.f=当前层的输入来自哪一层的输出  s的m.f都是-1
            if m.f != -1:  # if not from previous layer
                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile: # 打印日志信息  前向推理时间
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run 前向推理
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        """用在上面的__init__函数上
                将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
                de-scale predictions following augmented inference (inverse operation)
                :params p: 推理结果
                :params flips:
                :params scale:
                :params img_size:
                """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    # 打印日志信息  前向推理时间
    def _profile_one_layer(self, m, x, dt):
        #c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        c = isinstance(m, Detect) or isinstance(m, TDetect)  # is final layer, copy input as inplace fix

        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))

        # 以下为新增代码 ------------- 以上为原有代码🚀
        elif isinstance(m, TDetect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)

        return self

# 将在model模块的__init__函数中调用
def parse_model(d, ch):  # model_dict, input_channels(3)
    """用在上面Model模块中解析模型文件(字典形式)，并搭建网络结构
        这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel）=>使用当前层的参数搭建当前层 =>生成 layers + save
        :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
        :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
        :return nn.Sequential(*layers): 网络的每一层的层结构
        :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]这些层需要保存
        """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: number of anchors 每一个predict head上的anchor数 = 3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: number of outputs 每一个predict head层的输出channel = anchors * (classes + 5(xywhc)) = 75(VOC)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel/记录下一层的输入channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # from(当前层输入来自哪些层), number(当前层重复次数), module(当前层类别), args(当前层类参数)
    # 两个列表相加是类似字符串拼接那样进行拼接
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args 遍历backbone和head的每一层
        m = eval(m) if isinstance(m, str) else m  # eval strings ， eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
        # 没什么用
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # ------------------- 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
        # depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度) round()四舍五入
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain    n：该层重复多少次
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3STR, C3SPP, C3Ghost, ASPP, CBAM, nn.ConvTranspose2d, CoorAttention, CABottleneck, C3CA, SPPCSPC, SPPFCSPC, SCConv, HorBlock, C3HB, GnConv]:
            # c1: 当前层的输入的channel数  c2: 当前层的输出的channel数(初定)  ch: 记录着所有层的输出channel
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output, no=75  只有最后一层c2=no  最后一层不用控制宽度，输出channel必须是no
                # width gain 控制宽度  如v5s: c2*0.5  c2: 当前层的最终输出的channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)# 此函数在util中定义 8 是判断前面通道数是不是8的倍数
            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            # [in_channel, out_channel, *args[1:]]实际的输入通道 输出通道 ks stride
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost, 则需要在args中加入bottleneck的个数
            # [in_channel, out_channel, Bottleneck的个数n, bool(True表示有shortcut 默认，反之无)]
            if m in [BottleneckCSP, C3, C3TR, C3STR, C3Ghost, C3CA, C3HB, BAM]:
                args.insert(2, n)  # number of repeats # 在第二个位置插入bottleneck个数n
                n = 1 #如果不修改N=1，那么在后面构建模型的时候，该模块就会重复N次，但是这个N代表的是该模块内部组件的重复个数
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        elif m in [Concat, AdConcat2, AdConcat3]:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f) #通道数相加

        elif m in [ConvMix, CSPCM]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not outputss
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]

        elif m in [AdaptConcat, AdaptADD]:
            # AdaptConcat层则将f中所有的输出自适应累加得到这层的输出channel
            c2 = sum(ch[x] for x in f) #通道数相加
            level = len(f)
            args = [level, *args]  # 有几层 哪个维度上相加 每层的维度
        elif m in [Adapt_Add2, Adapt_Add3]:
            c2 = max([ch[x] for x in f])

        elif m in [C3GhostV2]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not outputss
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [C3GhostV2]:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is Detect: # Detect（YOLO Layer）层
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors 几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is TDetect:
            args.append([ch[x] for x in f])

        elif m is Contract:# 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:# 不怎么用
            c2 = ch[f] // args[0] ** 2
        elif m is space_to_depth:
            c2 = 4 * ch[f]
        elif m is SMMConv:
            c1 = ch[f]
            c2 = 4 * args[0]
            args = [c1, args[0]]
        elif m is DMMConv:
            c1 = ch[f]
            c2 = 5 * args[0]
            args = [c1, args[0]]
        elif m is DMMConv2:
            c1 = ch[f]
            c2 = args[0] + 4 * c1
            args = [c1, args[0]]
        elif m is DMConv:
            c1 = ch[f]
            c2 = 4 * args[0]
            args = [c1, args[0]]
        else:
            # Upsample
            c2 = ch[f] # args不变
        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m 加*就是为了传进去不是一个list,而是各个值
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module 往模型中传入参数
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type, t = module type   'modules.common.Focus'
        np = sum(x.numel() for x in m_.parameters())  # number params计算这一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_) # 将当前层结构module加入layers中
        if i == 0:
            ch = []# 去除输入channel [3]
        ch.append(c2)# 把当前层的输出channel数加入ch [32] [32, 64] [32, 64, 128]
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='C3CASPD2.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
