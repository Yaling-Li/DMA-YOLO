import os
import pandas as pd
from PIL import Image

YOLO_LABELS_PATH = "/data/lyl_data/VisDrone/VisDrone2019-DET-train/labels"
VISANN_PATH = "/data/lyl_data/VisDrone/VisDrone2019-DET-train/annotations"
VISIMG_PATH = "/data/lyl_data/VisDrone/VisDrone2019-DET-train/images"

#转换为yolo.txt格式,x,y,w,h分别为相对大小
def convert(bbox, img_size):
    #将标注visDrone数据集标注转为yolov5
    #bbox top_left_x top_left_y width height
    dw = 1/(img_size[0])
    dh = 1/(img_size[1])
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    #x,y代表gt的中心点坐标
    x = x * dw
    y = y * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    #x,y为对象的中相对其所在grid的中的位置
    #w,h表示真实的边界框的宽度和高度相对于整个图片的比例
    return (x,y,w,h) 

def ChangeToYolo5():
    if not os.path.exists(YOLO_LABELS_PATH):
        os.makedirs(YOLO_LABELS_PATH)
    print(len(os.listdir(VISANN_PATH)))#打印该文件夹下有多少张图像信息
    for file in os.listdir(VISANN_PATH):
        image_path = VISIMG_PATH + '/' + file.replace('txt', 'jpg') #图像
        ann_file = VISANN_PATH + '/' + file #每张图像对应的注释信息
        out_file = open(YOLO_LABELS_PATH + '/' + file, 'w')
        bbox = pd.read_csv(ann_file,header=None).values
        img = Image.open(image_path)
        img_size = img.size #获取图像的（宽，高）
        for row in bbox:
            if(row[4]==1 and 0<row[5]<11):   #属于object，并且类别不属于ignored regions和others，转换label
                label = convert(row[:4], img_size)
                out_file.write(str(row[5]-1) + " " + " ".join(str(f'{x:.6f}') for x in label) + '\n')
        out_file.close()

if __name__ == '__main__':
    ChangeToYolo5()