# #################查看模型 的 anchor  #######################
import torch
from models.experimental import attempt_load
model = attempt_load("/home/lg/tph-yolov5-main/runs/train/enhance6/weights/best.pt")
#model = attempt_load("/home/lg/tph-yolov5-main/runs/train/enhance6/weights/best.pt", map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(m.anchor_grid)

