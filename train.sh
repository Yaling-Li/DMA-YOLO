python train.py --img 1536 --adam --batch 4 --epochs 200 --data ./data/VisDrone.yaml --weights '' --hy data/hyps/hyp.VisDrone.yaml --cfg models/ablation-ca-scconv-sppfcspc.yaml --name ablation-ca-scconv-sppfcspc

CUDA_VISIBLE_DEVICES=1 python train.py --img 1024 --adam --batch 8 --epochs 150 --data ./data/UAVDT.yaml --weights "/home/lg/tph-yolov5-main/weights/yolov5l.pt" --hyp data/hyps/hyp.scratch.yaml --cfg models/C3CASPD2.yaml --name C3CASPD2-UAVDT

CUDA_VISIBLE_DEVICES=1 python train.py --img 1536 --adam --batch 4 --epochs 200 --data ./data/VisDrone.yaml --weights '/home/lg/tph-yolov5-main/weights/yolov5l.pt' --hy data/hyps/hyp.VisDrone.yaml --cfg models/CASPD_ODRTA.yaml --name CASPD_ODRTA --assignment tal
