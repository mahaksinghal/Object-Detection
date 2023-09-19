import cv2
from gui_buttons import Buttons
#import numpy as np

#Initialize Buttons
button=Buttons()
button.add_button("person",10,20)
button.add_button("laptop", 10, 80)
button.add_button("bottle", 10, 210)

colors = button.colors

# opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model =cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

#load class lists
classes = []
with open ("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("objects list")
print(classes)

#initialize the camera
cap=cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2600)
#full hd 1920*1880



def click_button(event, x, y, flags,params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
       # print(x,y)
         button.button_click(x,y)
      #  polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])

       # is_inside = cv2.pointPolygonTest(polygon,(x,y),False)
       # if is_inside>0:
        #    print("we are clicking inside the button",x,y)

          #  if button_person is False:
           #     button_person = True
           # else:
            #    button_person = False

          #  print("now button person is:",button_person)


#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)

while True:
    #get frames
    ret, frame = cap.read()

    #get active button list
    active_buttons = button.active_buttons_list()
    print("active buttons",active_buttons)

    #object detection
    (class_ids,scores,bboxes) = model.detect(frame)
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h)=bbox
        class_name = classes[class_id]

        if class_name in active_buttons:
          cv2.putText(frame , class_name, (x,y - 10), cv2.FONT_HERSHEY_PLAIN , 2,(100,0,50),2)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(150,0,50),3)

    #create button
    #cv2.rectangle(frame,(20,20),(220,70),(0,0,200),-1)
   # polygon = np.array([[(20,20),(220,20),(220,70),(20,70)]])
    #cv2.fillPoly(frame, polygon, (0,0,200))
   # cv2.putText(frame,"Person",(30,60),cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),3)

    #print ("class ids", class_ids)
    #print("scores",scores)
    #print("bboxes", bboxes)

    #display buttons
    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()