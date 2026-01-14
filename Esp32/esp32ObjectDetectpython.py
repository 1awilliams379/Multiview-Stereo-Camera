import cv2
import numpy as np
import urllib.request

url = 'http://10.0.0.107/cam-hi.jpg'

cap = cv2.VideoCapture(url)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []

    # Process each detection
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Draw boxes on image
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    # Step 1: Capture image
    success, img = cap.read()

    # Handle connection issues
    if not success:
        cap.release()
        cap = cv2.VideoCapture(url)
        continue  # Skip to next iteration

    # Step 2: Preprocess for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)

    # Step 3: Feed to network
    net.setInput(blob)

    # Step 4: Run forward pass
    layernames = net.getLayerNames()
    outputNames = [layernames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # Step 5: Process results and draw boxes
    findObject(outputs, img)

    # Step 6: Display
    cv2.imshow('Image', img)

    # Step 7: Check for quit (press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
