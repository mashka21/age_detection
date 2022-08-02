# Age Detection using OpenCV in jupyter google colab

step-1  

    !git clone https://github.com/mashka21/age_detection.git
    %cd age_detection

step-2  Downloading pretrained data and unzipping it

    # Downloading pretrained data and unzipping it
    !gdown https://drive.google.com/uc?id=1GyNXSbaLBMOYwGJw9726Xn0EWe4VJWlO
    # https://drive.google.com/uc?id=1GyNXSbaLBMOYwGJw9726Xn0EWe4VJWlO
    !unzip ageDetection.zip
    
step-3  Import required modules

    # Import required modules
    import cv2 as cv
    import math
    import time
    from google.colab.patches import cv2_imshow

    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

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
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

    faceProto = "modelNweight/opencv_face_detector.pbtxt"
    faceModel = "modelNweight/opencv_face_detector_uint8.pb"

    ageProto = "modelNweight/age_deploy.prototxt"
    ageModel = "modelNweight/age_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    padding = 20

    def age_gender_detector(frame):
        # Read frame
        t = time.time()
        frameFace, bboxes = getFaceBox(faceNet, frame)
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = "{}".format(age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        return frameFace
        
        
step-4 Output

    input = cv.imread("image.jpg")
    output = age_gender_detector(input)
    cv2_imshow(output)
