from vitpose_infer import VitInference
import cv2 

vid = cv2.VideoCapture('kek.mp4')
model = VitInference('models/vitpose-b-multi-coco.engine',\
            yolo_path='./yolov5n.engine',tensorrt=True)
frame_counter =0

while True:
    ret,frame = vid.read()
    if ret:
        pts,tids,bboxes,drawn_frame,orig_frame= model.inference(frame,frame_counter)
        # writer.write(cv2.resize(drawn_frame,(1280,720)))
        cv2.imshow("Video",drawn_frame)
        cv2.waitKey(1)
        frame_counter+=1
    else:
        break
vid.release()
writer.release()  
cv2.destroyAllWindows()