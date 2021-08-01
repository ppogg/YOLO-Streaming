import cv2 as cv
import numpy as np

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "data/yolo-fastest-xl.cfg"
modelWeights = "model/yolo-fastest-xl.weights"
classesFile = "data/coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
fastest_net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
fastest_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
fastest_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(color, classId, conf, left, top, right, bottom, frame):
    cv.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
    cv.putText(frame, label, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, confThreshold, nmsThreshold):
    frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        color = [int(c) for c in colors[classIds[i]]]
        drawPred(color, classIds[i], confidences[i], left, top, left + width, top + height, frame)

def v3_inference(frame):
    inpWidth, inpHeight = 320, 320
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), [0, 0, 0], swapRB=False, crop=False)
    # Sets the input to the network
    fastest_net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = fastest_net.forward(getOutputsNames(fastest_net))
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, confThreshold = 0.5, nmsThreshold = 0.2)
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = fastest_net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    print(label)

if __name__ == '__main__':
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(3, 960)  # set video width
    cap.set(4, 820)  # set video height
    while True:
        ret, frame = cap.read()
        v3_inference(frame)
        cv.imshow('fourcc', frame)
        k = cv.waitKey(20)
        # q键退出
        if (k & 0xff == ord('q')):
            break
    cap.release()
    cv.destroyAllWindows()
