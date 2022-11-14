# ViTPose-Pytorch no mmcv needed

ViTPose (https://github.com/ViTAE-Transformer/ViTPose) + ByteTrack (https://github.com/ifzhang/ByteTrack) + yolov5 (https://github.com/ultralytics/yolov5)
# Preparation
Download the official weights from ViTPose. Create a dictionary called models.
Tested with both the vitpose-b models.
```
mkdir models
```
```
mv <weight_path> ~/ViTPose-Pytorch/models/
```
# Requirements
Not sure if all of them are needed
```
pip3 insltall -r requirements.txt
```
# How to run
```
python3 inference_yolov5.py --path=<path_to_vid>
```

# TRT support
TRT 7.x is preferred, since TRT 8.x does not produce good results in my case.
```
pip3 install nvidia-pyindex nvidia-tensorrt==7.2.3.4
```
Export ViTPose TRT model 
```
python3 export.py --include engine --device 0
```
Export yolov5 engine, either by cloning the official yolov5 repo and using the export function or via cd /.cache/torch/hub/yolov5_master and 
exporting from there. Move the engine file into the root folder of ViTPose-Pytorch.
Run TRT infer on video

```
python3 inference_yolov5.py --path=<path_to_vid> --trt
```
# With a webcam or another device
```
ls /dev/vid*
```
Select the device you want to capture video with (0 by default if no arg is passed).

```
python3 inference_yolov5.py --camid=<id> (--trt)
```

Single Image inference. The output will be stored in test_out.png

```
python3 inference_yolov5.py --img=<img_path> (--trt)
```
