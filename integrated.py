# Import necessary libraries
import cv2
import numpy as np

# Function to preprocess input frame
def preprocess_frame(frame):
    # Perform preprocessing such as resizing, normalization, etc.
    # Return preprocessed frame
    return frame

# Function to detect humans using Model A
def detect_human(frame):
    import cv2
    import pandas as pd
    from ultralytics import YOLO

    model = YOLO('yolov8s.pt')

    def RGB(event, x, y, flags, param):
        if(event == cv2.EVENT_MOUSEMOVE):
            point = [x, y]
            print(point)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    output = cv2.VideoWriter('output_final.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (1020, 500))

    file = open('/Users/dhruvaggarwal/Downloads/PeopleCounting-ComputerVision-master/coco.names', 'r')
    data = file.read()
    class_list = data.split('\n')
    count=0
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    num_persons = 0

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            num_persons += 1
    img2 = cv2.resize(frame,(640,480))
    cv2.putText(img2, f'Number of Persons: {num_persons}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('RGB', img2)
    output.write(img2)

    return num_persons  # Dummy human count

# Function to detect weapons using Model B
def detect_weapon(frame):
    import cv2
    import numpy as np


# Load Yolo

    net = cv2.dnn.readNet("/Users/dhruvaggarwal/Downloads/yolov3_training_2000.weights", "/Users/dhruvaggarwal/Downloads/weapon-detection-python-opencv-withyolov5--main/yolov3_testing.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    classes = ["Weapon"]
    img = frame
    height, width, channels = img.shape
    # width = 512
    # height = 512

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    ans2=False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                ans2 = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    if indexes == 0: print("weapon detected in frame")
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img,(640,480))
    cv2.imshow("Image", img)
    return ans2  # Dummy weapon detection result

# Function to detect fire using Model C
def detect_fire(frame):
    from ultralytics import YOLO
    import cvzone
    import cv2
    import math
    # cap = cv2.VideoCapture("/Users/dhruvaggarwal/Downloads/pexels_videos_2646933 (240p).mp4")
    model = YOLO('/Users/dhruvaggarwal/Downloads/fire.pt')
    # Reading the classes
    classnames = ['fire']
    frame = cv2.resize(frame,(640,480))
    result = model(frame,stream=True)
    ans = False
    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                ans = True
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)

    cv2.imshow('frame',frame)

    return ans # Dummy fire detection result



# Main function for video surveillance
def main():
    # Open video capture device (replace 'video_file.mp4' with actual video source)
    cap = cv2.VideoCapture('/Users/dhruvaggarwal/Downloads/8625861-sd_640_360_25fps.mp4')

    while cap.isOpened():
        # Read frame from video stream
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Detect humans using Model A
        human_count = detect_human(preprocessed_frame)
        
        # Detect weapons using Model B
        weapon_detected = detect_weapon(preprocessed_frame)
        
        # Detect fire using Model C
        fire_detected = detect_fire(preprocessed_frame)
        
        # Perform decision logic based on detection results
        if weapon_detected:
            print("Weapon detected! Alert security.")
        if fire_detected:
            print("Fire detected! Evacuate affected areas.")
        
        # Display the frame (for demonstration purposes)
        # cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture device and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
