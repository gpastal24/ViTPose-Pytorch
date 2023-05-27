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
If nvidia-tensorrt is not found install nvidia-pyindex first.

```
pip3 insltall -r requirements.txt
```
# How to run

Export ViTPose TRT model 
```
python3 export.py --include engine --device 0
```
Export yolov5 engine, either by cloning the official yolov5 repo and using the export function or via cd /.cache/torch/hub/yolov5_master and 
exporting from there. Move the engine file into the root folder of ViTPose-Pytorch.

# With a webcam or another device
```
ls /dev/vid*
```
3 demo examples are provided . To use this lib as a package run 

```
python3 -m build
```
Install the whl in the dist folder via pip.


# Examples 

![](https://github.com/gpastal24/ViTPose-Pytorch/blob/main/examples/output_simple.gif)
With TRT
![](https://github.com/gpastal24/ViTPose-Pytorch/blob/main/examples/trt_out.gif)

![](https://github.com/gpastal24/ViTPose-Pytorch/blob/main/examples/out.jpg?raw=true)
