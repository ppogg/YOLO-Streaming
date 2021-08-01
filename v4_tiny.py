import cv2
import numpy as np

model = 'model/yolov4-tiny.weights'
config = r'data/yolov4-tiny.cfg'
labelsPath = r"data/coco.names"

v4tiny_Net = cv2.dnn.readNetFromDarknet(config, model)
ln = v4tiny_Net.getLayerNames()
ln = [ln[i[0] - 1] for i in v4tiny_Net.getUnconnectedOutLayers()]

LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

def postprocess(frame, networkOutput, conf, threshold):
    (H, W) = frame.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    for output in networkOutput:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            # end = time.time()
            # seconds = end - begin
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

def v4_inference(frame):
    conf, threshold = 0.4, 0.4
    v4tiny_Net.setInput(cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False))
    networkOutput = v4tiny_Net.forward(ln)
    postprocess(frame, networkOutput, conf, threshold)
    # localtime = time.asctime(time.localtime(time.time()))
    t, _ = v4tiny_Net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    # draw_time(frame, t)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 960)  # set video width
    cap.set(4, 720)  # set video height
    while True:
        # begin = time.time()
        ret, frame = cap.read()
        v4_inference(frame)
        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
