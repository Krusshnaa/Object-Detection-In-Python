import cv2

def Camera():
    cam = cv2.VideoCapture(1) // To Capture Webcam 

    cam.set(3, 740)
    cam.set(4, 580)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320 , 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        success, img = cam.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Write the frame into the file 'output.avi'
        out.write(img)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cam.release()
    out.release()
    cv2.destroyAllWindows()

Camera()





# # facebook: https://www.facebook.com/salar.brefki/
# # instagram: https://www.instagram.com/salarbrefki/

# import cv2

# ####### From Image #######
# def ImgFile():
#    img = cv2.VideoCapture(0)

#    classNames = []
#    classFile = 'coco.names'

#    with open(classFile, 'rt') as f:
#       classNames = f.read().rstrip('\n').split('\n')

#    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#    weightpath = 'frozen_inference_graph.pb'

#    net = cv2.dnn_DetectionModel(weightpath, configPath)
#    net.setInputSize(320 , 230)
#    net.setInputScale(1.0 / 127.5)
#    net.setInputMean((127.5, 127.5, 127.5))
#    net.setInputSwapRB(True)

#    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
#    print(classIds, bbox)

#    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#       cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#       cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
#                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)


#    cv2.imshow('Output', img)
#    cv2.waitKey(0)
# ######################################

# ####### From Video or Camera #######
# def Camera():
#    cam = cv2.VideoCapture(0)

#    cam.set(3, 740)
#    cam.set(4, 580)

#    classNames = []
#    classFile = 'coco.names'

#    with open(classFile, 'rt') as f:
#       classNames = f.read().rstrip('\n').split('\n')

#    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#    weightpath = 'frozen_inference_graph.pb'

#    net = cv2.dnn_DetectionModel(weightpath, configPath)
#    net.setInputSize(320 , 230)
#    net.setInputScale(1.0 / 127.5)
#    net.setInputMean((127.5, 127.5, 127.5))
#    net.setInputSwapRB(True)

#    while True:
#       success, img = cam.read()
#       classIds, confs, bbox = net.detect(img, confThreshold=0.5)
#       print(classIds, bbox)

#       if len(classIds) !=0:
#          for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#             cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#             cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)


#       cv2.imshow('Output', img)
#       cv2.waitKey(1)
# ######################################


# ## Call ImgFile() Function for Image Or Camera() Function for Video and Camera
# # ImgFile()
# Camera()
