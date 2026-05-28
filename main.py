import cv2
import math
import argparse

# 1. Face detection ke liye function (Jesa aapke 2nd screenshot mein hai)
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
    # Image ko blob mein convert karna model input ke liye
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # Face ke charo taraf rectangle draw karna
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
            
    return frameOpencvDnn, bboxes

# 2. Command Line Arguments setup (Jesa aapke 3rd screenshot mein hai)
parser = argparse.ArgumentParser()
parser.add_argument("-i", help='Give image file path (Optional)')
args = parser.parse_args()

# 3. Parameters Define Karna (Jesa aapke 4th screenshot mein hai)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# 4. Model paths assign karna (Jesa aapke 5th screenshot mein hai)
faceProto = "opencv_face_detector.txt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.txt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.txt"
genderModel = "gender_net.caffemodel"

# 5. Models ko Load karna (Jesa aapke 6th screenshot mein hai)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

# 6. Input Source Select Karna (Image path diya hai ya Webcam use karna hai)
# Agar aapne command line me -i ke sath image nahi di, toh webcam (0) start hoga
cap = cv2.VideoCapture(args.i if args.i else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
        
    # Face detect karna
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    if not bboxes:
        print("No face detected in this frame.")
        
    for bbox in bboxes:
        # Face area ko crop (extract) karna detection ke liye
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                     
        # Gender and Age models ke liye blob banana
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Gender Prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f"Gender: {gender}, Confidence: {genderPreds[0].max():.2f}")
        
        # Age Prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f"Age: {age}, Confidence: {agePreds[0].max():.2f}")
        
        # Screen par text display karne ke liye formatting
        label = f"{gender}, {age}"
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
    # Output Window Show karna
    cv2.imshow("Age and Gender Prediction", frameFace)

cap.release()
cv2.destroyAllWindows()