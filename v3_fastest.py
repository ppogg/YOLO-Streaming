import cv2 as cv
import argparse
import numpy as np

# Initialize the parameters
confThreshold = 0.25  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "data/yolo-fastest-xl.cfg"
modelWeights = "data/yolo-fastest-xl.weights"
# Load names of classes
classesFile = "data/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
np.random.seed(42)
colors = []
colors = np.random.randint(0, 255, size=(len(classes), 3),
                           dtype="uint8")
fastest_net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
fastest_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
fastest_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # print(dir(net))
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(color, classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    # cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
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
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
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

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        color = [int(c) for c in colors[classIds[i]]]
        drawPred(color, classIds[i], confidences[i], left, top, left + width, top + height, frame)

def v3_inference(frame):
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), [0, 0, 0], swapRB=False, crop=False)
    # Sets the input to the network
    fastest_net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = fastest_net.forward(getOutputsNames(fastest_net))
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = fastest_net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (8, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
    parser.add_argument('--image', type=str, default='', help='Path to image file.')
    parser.add_argument('--fourcc', type=int, default=1, help='Open the videocapture')
    parser.add_argument('--video', type=str, default='', help='Open the video')
    args = parser.parse_args()

    # Process inputs
    if args.fourcc == 0:
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

    elif args.image:
        frame = cv.imread(args.image)
        # Create a 4D blob from a frame.
        v3_inference(frame)
        cv.imshow('image', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        cap = cv.VideoCapture(args.video)
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        out = cv.VideoWriter('yolo-faster.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        font = cv.FONT_HERSHEY_SIMPLEX
        while (cap.isOpened()):
            ret, frame = cap.read()
            v3_inference(frame)
            cv.imshow('video', frame)
            out.write(frame)
            k = cv.waitKey(20)
            # q键退出
            if (k & 0xff == ord('q')):
                break
        cap.release()
        cv.destroyAllWindows()
