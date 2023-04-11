# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640

    这个文件主要是通过train.py调用run函数，而不是执行val.py。所以整个脚本最重要的是run函数
    用于每一轮训练结束后，验证当前模型的map、混淆矩阵等指标。
"""

import argparse  # 解析命令行参数模块
import json # 实现字典列表和JSON字符串之间的相互解析
import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread    # 线程操作模块

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve() # FILE = WindowsPath 'F:\yolo_v5\yolov5-U\val.py'
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 将'F:/yolo_v5/yolov5-U'加入系统的环境变量  该脚本结束后失效
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad() #不参与反向传播
#这里的run函数其实是train在执行，而不是通过val.py执行，用于每次训练epoch后验证当前模型
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Initialize/load model and set device
    # 初始化模型并选择相应的计算设备
    # 判断是否是训练时调用run函数(执行train.py脚本), 如果是就使用训练时的设备 一般都是train
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    # 如果不是trin.py调用run函数(执行val.py脚本)就调用select_device选择可用的设备
    # 并生成save_dir + make dir + 加载模型model + check imgsz + 加载data配置信息
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories# 生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model 加载模型 load FP32 model  只在运行val.py才需要自己加载model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # gs: 模型最大的下采样stride 一般[8, 16, 32] 所有gs一般是32
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        # 检测输入图片的分辨率imgsz是否能被gs整除 只在运行test.py才需要自己生成check imgsz
        # imgsz_test
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        data = check_dataset(data)  # check
    # ============================================== 2、调整模型设置 ==================================================
    # Half只能在单GPU设备上才能使用
    # 一旦使用half, 不但模型需要设为half, 输入模型的图片也需要设为half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
    model.eval()
    # Configure  启动模型验证模式
    # ============================================== 3、初始化配置2 ==================================================
    # 测试数据是否是coco数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # mAP@0.5:0.95 iou个数=10个
    niou = iouv.numel()

    # ============================================== 4、加载val数据集 ==================================================
    # 如果不是训练(执行val.py脚本调用run函数)就调用create_dataloader生成dataloader
    # 如果是训练(执行train.py调用run函数)就不需要生成dataloader 可以直接从参数中传过来testloader
    if not training:
        if device.type != 'cpu':
            # 这里创建一个全零数组测试下前向传播是否能够正常运行
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 创建dataloader 这里的rect默认为True 矩形推理用于测试集 在不影响mAP的情况下可以大大提升推理速度
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]
    # ============================================== 5、初始化配置3 ==================================================
    # 初始化一些测试需要的参数
    seen = 0 # 初始化测试的图片的数量
    # 初始化混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 获取coco数据集的类别索引
    # coco数据集是80个类 索引范围本应该是0~79,但是这里返回的确是0~90  coco官方就是这样规定的
    # coco80_to_coco91_class就是为了与上述索引对应起来，返回一个范围在0~80的索引数组
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    #s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    # 初始化p, r, f1, mp, mr, map50, map指标和时间t0, t1, t2
    #dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt, p, r, f1, mp, mr, map50, map75, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件中的字典 统计信息 ap等
    jdict, stats, ap, ap_class = [], [], [], []
    # ============================================== 6、开始验证 ==================================================
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 6.1、预处理图片和target
        t1 = time_sync()# 获取当前时间
        img = img.to(device, non_blocking=True)
        # 如果half为True 就把图片变为half精度  uint8 to fp16/32
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        # 6.2、Run model  前向推理
        # out:       推理结果 1个 [bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
        # train_out: 训练结果 3个 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
        #                    如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        out, train_out = model(img, augment=augment)  # inference and training outputs
        dt[1] += time_sync() - t2# 累计前向推理时间 t1

        # Compute loss
        # 6.3、计算验证集损失
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        # 6.4、Run NMS
        # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # save_hybrid: adding the dataset labels to the model predictions before NMS
        #              是在NMS之前将数据集标签targets添加到模型预测中
        # 这允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        #ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP@0.5:0.95
        #mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    #pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    #LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            #LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            #map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            map, map50, map75 = eval.stats[:3]  # update results (mAP@0.5:0.95, mAP@0.75, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    #return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """
       opt参数详解
       data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
       weights: 模型的权重文件地址 weights/yolov5s.pt
       batch_size: 前向传播的批次大小 默认32
       imgsz: 输入网络的图片分辨率 默认640
       conf-thres: object置信度阈值 默认0.001
       iou-thres: 进行NMS时IOU的阈值 默认0.6
       task: 设置测试的类型 有train, val, test, speed or study几种 默认val
       device: 测试的设备
       single-cls: 数据集是否只用一个类别 默认False
       augment: 测试是否使用TTA Test Time Augment 默认False
       verbose: 是否打印出每个类别的mAP 默认False
       下面三个参数是auto-labelling(有点像RNN中的teaching forcing)相关参数详见:https://github.com/ultralytics/yolov5/issues/1563 下面解释是作者原话
       save-txt: traditional auto-labelling
       save-hybrid: save hybrid autolabels, combining existing labels with new predictions before NMS (existing predictions given confidence=1.0 before NMS.
       save-conf: add confidences to any of the above commands
       save-json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
       project: 测试保存的源文件 默认runs/test
       name: 测试保存的文件地址 默认exp  保存在runs/test/exp下
       exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
       half: 是否使用半精度推理 默认False
       """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    """这个模块根据opt.task可以分为三个分支，我们主要的分支还是在 opt.task in (‘train’, ‘val’, ‘test’)。而其他的两个分支，大概看看在干什么就可以了，没什么用。一般都是直接进入第一个分支，执行run函数."""
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
