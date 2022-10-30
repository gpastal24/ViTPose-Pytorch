from builder import build_model
import torch
import ssl
import cv2
from tracker import byte_tracker
from visualizer import plot_tracking
import argparse

from torch2trt import torch2trt, TRTModule
from pose_viz import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from pose_utils import pose_points_res18,pose_points,pose_points_yolo5,check_pose_weights,convert_pose
from timerr import Timer
from general_utils import polys_from_pose
# ssl._create_default_https_context = ssl._create_unverified_context ##Bypass certificate has expired error for yolov5
import logging
from logger_helper import CustomFormatter
import numpy as np




logger = logging.getLogger("Tracker !")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.propagate=False

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
   

    parser.add_argument(
        "--path", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # exp file

    parser.add_argument('--trt',action='store_true')
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=240, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=15, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser
# check_pose_weights()
# convert_pose()
args = make_parser().parse_args()
def convert_to_trt(net, output_name, height, width):
                net.eval()
                img = torch.randn( 1,3, height, width).cuda()
                # img = img.cuda()
                print('Starting conversion')
                net_trt = torch2trt(net, [img],max_batch_size=10,fp16_mode=True)
                torch.save(net_trt.state_dict(), output_name)
                print('Conversion Succesufl!')
tracker = byte_tracker.BYTETracker(args,frame_rate=30)
if args.trt:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.engine')

    from ViTPose_trt import TRTModule_ViTPose
    
    pose = TRTModule_ViTPose(path='models/vitpose-b-multi-coco.engine',device='cuda:0')
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

    pose = build_model('ViTPose_base_coco_256x192','./models/vitpose-b.pth')
pose.cuda().eval()
# args = make_parser
if args.path is not None:

    vid = cv2.VideoCapture(args.path)
else:
    vid = cv2.VideoCapture(args.camid)

frame_id = 0
timer = Timer()
timer_track = Timer()
timer_det = Timer()

while(True):
    timer.tic()
    ret, frame = vid.read()
    if ret:

        frame_orig = frame.copy()
        # timer_det.tic()
        pts,online_tlwhs,online_ids,online_scores = pose_points_yolo5(model, frame, pose, tracker,args)
        # timer_det.toc()
        # logger.info("FPS detector = %s",1./timer_det.average_time)
        
        # dets = res.xyxy[0]
        # dets = dets[dets[:,5] == 0.]

        if len(online_ids)>0:
            # timer_track.tic()
            timer.toc()
            # print(online)
            # logger.info(online_tlwhs)
            # logger.info(online_ids)
            online_im = plot_tracking(
                frame_orig, online_tlwhs, online_ids, frame_id=frame_id, fps=1/timer.average_time
            )
            # person_ids = np.arange(len(pts), dtype=np.int32)
            if pts is not None:
                for i, (pt, pid) in enumerate(zip(pts, online_ids)):
                    online_im=draw_points_and_skeleton(online_im, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.3)
            # seg = polys_from_pose(pts)
            # print(seg)
            # for s in seg:
            #     poly = s
            #     # make a new list from every 2 elements of list polys
            #     # print(poly)
            #     composite_list = [poly[x:x + 2] for x in range(0, len(poly), 2)]
            #     # print(composite_list)
            #     pts = np.array(composite_list)
            #     # print(pts)
            #     pts = pts.astype(np.int32)
            #     pts = pts.reshape((-1, 1, 2))
            #     cv2.polylines(online_im, [pts], True, (255, 0, 255), 2)

            # online_im = 
        else:
            timer.toc()
            online_im = frame_orig


        cv2.imshow('frame',online_im)
        frame_id+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()

