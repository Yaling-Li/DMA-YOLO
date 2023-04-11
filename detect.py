# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

#这部分的主要作用有两个：1、将当前项目添加到系统路径上，使得项目中的模块可以调用2、将当前项目额相对路径保存到root中，便于寻找项目中的文件
FILE = Path(__file__).resolve()#__file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directoryROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:# sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PAT 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  ROOT设置为相对路径

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()# 该标注使得方法中所有计算得出的tensor的requires_grad都自动设置为False，也就是说不会求梯度，可以加快预测效率，减小资源消耗
#设置参数的读入
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s) 事先训练完成的权重文件，比如yolov5s.pt,假如使用官方训练好的文件（比如yolov5s）,则会自动下载
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam预测时的输入数据，可以是文件/路径/URL/glob, 输入是0的话调用摄像头作为输入
        imgsz=640,  # inference size (pixels)预测时的放缩后图片大小(因为YOLO算法需要预先放缩图片)
        conf_thres=0.25,  # confidence threshold置信度阈值, 高于此值的bounding_box才会被保留
        iou_thres=0.45,  # NMS IOU thresholdIOU阈值,高于此值的bounding_box才会被保留
        max_det=1000,  # maximum detections per image一张图片上检测的最大目标数量
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu所使用的GPU编号，如果使用CPU就写cpu
        view_img=False,  # show results是否在推理时预览图片
        save_txt=False,  # save results to *.txt是否将结果保存在txt文件中
        save_conf=False,  # save confidences in --save-txt labels是否将结果中的置信度保存在txt文件中
        save_crop=False,  # save cropped prediction boxes是否保存裁剪后的预测框
        nosave=False,  # do not save images/videos是否保存预测后的图片/视频
        classes=None,  # filter by class: --class 0, or --class 0 2 3过滤指定类的预测结果
        agnostic_nms=False,  # class-agnostic NMS# 如为True,则为class-agnostic. 否则为class-specific
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name推理结果保存路径
        name='exp',  # save results to project/name结果保存文件夹的命名前缀
        exist_ok=False,  # existing project/name ok, do not incrementTrue: 推理结果覆盖之前的结果 False: 推理结果新建文件夹保存,文件夹名递增
        line_thickness=3,  # bounding box thickness (pixels)绘制Bounding_box的线宽度
        hide_labels=False,  # hide labels  True: 隐藏标签
        hide_conf=False,  # hide confidences True: 隐藏置信度
        half=False,  # use FP16 half-precision inference是否使用半精度推理（节约显存）
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    #输入路径变为字符串
    source = str(source)
    #save_img：bool 判断是否要保存图片，如果nosave为false且source的结尾不是txt则保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    #判断文件是否为视频流
    #Path()提取文件名 例如：Path("./data/test_images/bus.jpg") Path.name->bus.jpg Path.parent->./data/test_images Path.suffix->.jpg
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)#提取文件后缀名是否符合要求的文件，例如：是否格式是jpg, png, asf, avi等
    #.lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写, .startswith('http://')返回True or Flase
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))#判断source是否为链接
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)#isnumeric()方法检查字符串是否仅由数字组成，判断source是否是摄像头
    if is_url and is_file:
        source = check_file(source)  # download，如果source是一个指向图片/视频的链接,则下载输入数据

    # Directories
    #创建本次推理的目录,不存在就新建，按照实验文件以此递增新建
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  #save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  #根据前面生成的路径创建文件夹

    # Initialize，判断是使用CPU还是使用GPU，使用GPU时可以选择使用半精度推理
    device = select_device(device)#select_device方法定义在utils.torch_utils模块中，返回值是torch.device对象，也就是推理时所使用的硬件资源。输入值如果是数字，表示GPU序号。也可是输入‘cpu’，表示使用CPU训练，默认是cpu
    half &= device.type != 'cpu'  # half precision only supported on CUDA half:bool

    # Load model加载模型
    w = str(weights[0] if isinstance(weights, list) else weights)#判断weights是否为列表，如果是，则加载weights[0]，否则加载weights，这里weight可以传入多个
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']#Path(w).suffix.lower()找出权重文件的后缀
    check_suffix(w, suffixes)  # check weights have acceptable suffix判断是否存在可以处理的权重文件
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]
    # assign defaults模型步长为64
    #names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...]

    #权重文件属于pt文件
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)#加载模型
        stride = int(model.stride.max())  # model stride模型的最大步长
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names类别（cls）的名字，hasattr() 函数用于判断对象是否包含对应的属性
        #使用半精度，默认不使用
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier 加载的分类模型，先检测目标框，再进行分类，默认是不使用的
            modelc = load_classifier(name='resnet50', n=2)  # initialize 模型加载
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    #权重文件属于onnx文件
    elif onnx:
        #使用dnn进行推理，默认不使用
        if dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            if "edgetpu" in w:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                import tflite_runtime.interpreter as tflri
                delegate = {'Linux': 'libedgetpu.so.1',  # install libedgetpu https://coral.ai/software/#edgetpu-runtime
                            'Darwin': 'libedgetpu.1.dylib',
                            'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = tflri.Interpreter(model_path=w, experimental_delegates=[tflri.load_delegate(delegate)])
            else:
                interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)
    # check image size
    #将图像大小调整为步长的整数倍，比如假如步长是10，imagesz是[100,101],则返回值是[100,100]

    # Dataloader加载数据
    #0使用摄像头作为输入
    if webcam:
        view_img = check_imshow()#检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        cudnn.benchmark = True  # set True to speed up constant image size inference该设置可以加速预测
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)#加载输入数据流
        #source：输入数据源 image_size 图片识别前被放缩的大小， stride：识别时的步长
        #auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要
        bs = len(dataset)  # batch_size批大小
    else:
        #直接从source文件下读取图片
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    #保存的路径？？？这里不太明白
    vid_path, vid_writer = [None] * bs, [None] * bs# 用于保存视频,前者是视频路径,后者是一个cv2.VideoWriter对象

    # Run inference进行推理
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once运行一次   next(model.parameters()表明数据是在CPU还是GPU上
    dt, seen = [0.0, 0.0, 0.0], 0
    #dt:存储每一步骤的耗时
    #seen：已经处理完多少帧图片
    for path, img, im0s, vid_cap, s in dataset:
    # 在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
    # path：文件路径（即source）
    # img: 处理后的输入图片列表（经过了放缩操作）
    # im0s: 源输入图片列表
    # vid_cap
    # s： 图片的基本信息，比如路径，大小
        t1 = time_sync()#时间同步，获取当前时间
        if onnx:
            img = img.astype('float32')#强制转换数据类型
        else:
            img = torch.from_numpy(img).to(device)#将图片放到指定设备（如GPU）上识别
            img = img.half() if half else img.float()  # uint8 to fp16/32 转换为半精度浮点数进行推理，把输入从整型转化为半精度/全精度浮点数。
        img /= 255  # 0 - 255 to 0.0 - 1.0将图片归一化处理（这是图像表示方法的的规范，使用浮点数就要归一化）
        if len(img.shape) == 3:#img的shape为宽，高，通道
            img = img[None]  # expand for batch dim再数组最后增加一个维度，为batchsize留位置添加一个第0维。在pytorch的nn.Module的输入中，第0维是batch的大小，这里添加一个1。
        t2 = time_sync()#时间同步 获取当前时间
        dt[0] += t2 - t1#计算本阶段所花费的时间

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False#可视化文件路径
            #如果为True则保留推理过程中的特征图，保存在runs文件夹中
            pred = model(img, augment=augment, visualize=visualize)[0]#这里加[0]是什么意思？？？可不可以不用
            #pred保留所有的bound_box的信息
            """
            pred.shape=(1, num_boxes, 5+num_class)  5是指x,y,w,h,conf
            h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """
        elif onnx:
            if dnn:
                net.setInput(img)
                pred = torch.tensor(net.forward())
            else:
                pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        #预测的时间
        dt[1] += t3 - t2

        # NMS 非极大值抑制
        """
        pred: 网络的输出结果
        conf_thres:置信度阈值
        ou_thres:iou阈值
        agnostic_nms: 进行nms是否也去除不同类别之间的框
        max-det: 保留的最大检测框数量
        ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor], 长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
        """
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # 执行非极大值抑制，返回值为过滤后的预测框
        # conf_thres： 置信度阈值
        # iou_thres： iou阈值
        # classes: 需要过滤的类（数字列表）
        # agnostic_nms： 标记class-agnostic或者使用class-specific方式。默认为class-agnostic
        # max_det: 检测框结果的最大数量

        dt[2] += time_sync() - t3#本阶段消耗的时间

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image  每次迭代处理一张图片
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #frame：此次取的是第几张图片
                s += f'{i}: '#s后面拼接一个字符串i
            else:
                # 大部分我们一般都是从LoadImages流读取本地文件中的照片或者视频 所以batch_size=1
                # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                # frame: 视频流
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 当前路径yolov5/data/images/
            #图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
            save_path = str(save_dir / p.name)  # img.jpg推理结果图片保存的路径
            #设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #设置打印图片的信息
            s += '%gx%g ' % img.shape[2:]  # print string显示推理前裁剪后的图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh得到原图的宽和高
            #保存截图
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 如果save_crop的值为true， 则将检测到的bounding_box单独保存成一张图片。
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            if len(det):
                # Rescale boxes from img_size to im0 size
                #将预测信息映射到原图
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）
                # Print results打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # 打印出所有的预测结果  比如1 person（检测出一个人）

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file保存txt文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 将坐标转变成x y w h 的形式，并归一化
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        #line的形式是： ”类别 x y w h“，假如save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”

                    if save_img or save_crop or view_img:  # Add bbox to image给图片添加推理后的bounding_box边框
                        c = int(cls)  # integer class类别编号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#类别名
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        #绘制边框
                        if save_crop:#将预测框内的图片单独保存
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)打印耗时
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results   im0是绘制好的图片
            im0 = annotator.result()
            if view_img:#如果为true，则显示该图片
                cv2.imshow(str(p), im0)#预览图片
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)如果为true，则保存绘制完的图片
            if save_img:
                if dataset.mode == 'image':#如果是图片，则保存
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'如果是视频或者“流”
                    if vid_path[i] != save_path:  # new video# vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                    #以上均为保存视频文件

    # Print results打印结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image平均每张图片所耗费的时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

#相关参数解释
def parse_opt():
    parser = argparse.ArgumentParser()#创建解释器
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)') #需要加载的权重
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')#需要进行推理的图片，可以是图片/视频路径，也可以是“0”（电脑自带摄像头），也可以是rtsp等视频流，默认'data/images'
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')#网络输入图片尺寸，莫仍大小为640
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')#置信度阈值，默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')#非极大值抑制NMS IOU阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')#最大侦测的目标数，每张图片中检测目标的个数最多为1000
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#设置设备CPU/CUDA，可以不用设置
    parser.add_argument('--view-img', action='store_true', help='show results')#是否展示推理后的图片/视频，默认为False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')#是否将预测的框坐标保存为txt，默认为false
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')#在保存的txt里面，除了类别，在保存对应的置信度
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')#是否保存裁剪预测框图片, 默认为False, 使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')#不保存图片/视频
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')#过滤得到为CLASSES分类的图片
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#不通类别间也可以做非极大值抑制（不开启的话，每个类别单独做NMS）
    parser.add_argument('--augment', action='store_true', help='augmented inference')#推理增强
    parser.add_argument('--visualize', action='store_true', help='visualize features')#可视化特征
    parser.add_argument('--update', action='store_true', help='update all models')#将模型中包含的优化器、ema等操作去除，减小模型的大小
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')#保存测试日志的文件夹路径
    parser.add_argument('--name', default='exp', help='save results to project/name')#保存测试日志文件夹的名字, 所以最终是保存在project/name中
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')# 是否重新创建日志文件, False时重新创建文件
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')#边界框厚度
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')#隐藏每个目标的标签
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')#隐藏每个目标的置信度
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')#FP16，半精度推理（增加推理速度）
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')#使用DNN作为ONNX的推理
    opt = parser.parse_args()#初始化解析器
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))#检查依赖包，内含可以不用满足的包
    run(**vars(opt))#运行run(),vars是将opt解释为字典格式，字典传参需要加**，列表传参需要加*

#主函数
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
