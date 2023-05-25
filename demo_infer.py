from src.vitpose_infer import VitInference
import cv2 

vid = cv2.VideoCapture('kek.mp4')
model = VitInference('models/vitpose-b-multi-coco.pth',\
            yolo_path='yolov5n.engine',tensorrt=False)
frame_counter =0

while True:
    ret,frame = vid.read()
    if ret:
        # print(ret)
        pts,tids,bboxes,drawn_frame,orig_frame= model.inference(frame,frame_counter)
        cv2.imshow("Video",drawn_frame)
        cv2.waitKey(1)
        frame_counter+=1
    else:
        break
vid.release()
# writer.release()  
cv2.destroyAllWindows()
