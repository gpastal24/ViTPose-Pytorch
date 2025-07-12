from src.vitpose_infer import VitInference
import cv2 

vid = cv2.VideoCapture('scenario7.avi')
model = VitInference('models/vitpose_base_whole.pth',\
            yolo_path='./yolov5n.pt',tensorrt=False,wholebody=True)

frame = cv2.imread('team.jpg')
pts,tids,bboxes,drawn_frame,orig_frame= model.inference(frame)
cv2.imwrite('out.jpg',drawn_frame)
