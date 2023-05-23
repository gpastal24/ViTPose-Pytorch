from src.vitpose_infer import VitInference
import cv2 

vid = cv2.VideoCapture('scenario7.avi')
model = VitInference('models/vitpose-b-multi-coco.engine',\
            yolo_path='./yolov5n.engine',tensorrt=True)

frame = cv2.imread('000000001000.jpg')
pts,tids,bboxes,drawn_frame,orig_frame= model.inference(frame)
cv2.imwrite('out.jpg',drawn_frame)
