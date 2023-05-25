from .model_builder import build_model
import torch
import ssl
import cv2
from .tracker import byte_tracker
from .pose_utils.visualizer import plot_tracking
import argparse

# from torch2trt import torch2trt, TRTModule
from .pose_utils.pose_viz import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from .pose_utils.pose_utils import pose_points_yolo5
from .pose_utils.timerr import Timer
from .pose_utils.general_utils import polys_from_pose,make_parser
# ssl._create_default_https_context = ssl._create_unverified_context ##Bypass certificate has expired error for yolov5
import logging
from .pose_utils.logger_helper import CustomFormatter
import numpy as np




logger = logging.getLogger("Tracker !")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.propagate=False

# def make_parser2():
#     parser = argparse.ArgumentParser("VitPose + yolov5 + Bytetrack demo!")
   

#     parser.add_argument(
#         "--path", help="path to video"
#     )
#     parser.add_argument("--path_out",default=None,help="save video")
#     parser.add_argument(
#         "--img", help="path to img",default=None
#     )
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

#     parser.add_argument('--trt',action='store_true')
#     parser.add_argument('--trt_pose_only',action='store_true')
#     # tracking args
#     return parser
# # check_pose_weights()
# # convert_pose()
# args = make_parser2().parse_args()

class VitInference:
    # tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
    def __init__(self,pose_path,yolo_path,tensorrt=False):
        super(VitInference,self).__init__()
        self.tensorrt = tensorrt
        self.tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
        self.pose_path = pose_path
        self.yolo_path = yolo_path
        print(self.pose_path)
        if self.tensorrt:
            assert self.yolo_path is not None
            pose_split = self.pose_path.split('.')[-1]
            assert pose_split =='engine'
            self.model = torch.hub.load('ultralytics/yolov5:v6.2', 'custom', yolo_path)

            from .pose_utils.ViTPose_trt import TRTModule_ViTPose
        
            self.pose = TRTModule_ViTPose(path=self.pose_path,device='cuda:0')
        else:
            
            self.model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5n', pretrained=True)
            self.pose = build_model('ViTPose_base_coco_256x192',self.pose_path)
        self.pose.cuda().eval()

        frame_id = 0
        self.timer = Timer()
        timer_track = Timer()
        timer_det = Timer()


            # frame = cv2.resize(frame,(640,360))
    def inference(self,img,frame_id=0):
        frame_orig = img.copy()
        self.timer.tic()
        pts,online_tlwhs,online_ids,online_scores = pose_points_yolo5(self.model, img, self.pose, self.tracker,self.tensorrt)

        self.timer.toc()
        if len(online_ids)>0:
            # timer_track.tic()
            # self.timer.tic()
            online_im = frame_orig.copy()
            online_im = plot_tracking(
                frame_orig, online_tlwhs, online_ids, frame_id=frame_id, fps=1/self.timer.average_time
            )
            # self.timer.toc()
            if pts is not None:
                for i, (pt, pid) in enumerate(zip(pts, online_ids)):
                    online_im=draw_points_and_skeleton(online_im, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.3)

        else:

            online_im = frame_orig
            
        return pts,online_ids,online_tlwhs,online_im,frame_orig



