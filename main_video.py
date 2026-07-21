import pickle 
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from SpotPicker import SAVE_FILE, BOX_WIDTH, BOX_HEIGHT



with open(SAVE_FILE, 'rb') as f:
    spots = pickle.load(f)

with open("./models/model_rfc.p",'rb') as f:
    model = pickle.load(f)

VIDEO_PATH = "carPark.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
spot_crops = []
valid_spots = []

while True:

    test, frame = cap.read()

    if not test or frame is None:
        break 

    frame_count += 1
    # Preprocess

    for x,y in spots:
        img = frame[y : y + BOX_HEIGHT, x : x + BOX_WIDTH]
        
        # Convert BGR (OpenCV) to RGB (skimage.io.imread format used in training)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to (50, 50) as done in training
        img = cv2.resize(img, (50, 50))

        # Normalize to [0, 1] range (skimage.transform.resize returns float values in [0, 1])
        img = img / 255.0

        img = img.flatten().reshape(1, -1)

        spot_crops.append(img.flatten())
        valid_spots.append((x, y))

    if frame_count % 30 == 0 or spot_crops:
        predictions = model.predict(np.array(spot_crops))
    
    for (x, y), pred in zip(valid_spots, predictions):

        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + BOX_WIDTH, y + BOX_HEIGHT), color, 2)
    
    cv2.imshow('Live Parking',frame)

    if cv2.waitKey(30) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
