import pickle 
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from SpotPicker import IMAGE_PATH , BOX_WIDTH , BOX_HEIGHT

from utils import get_fps , draw_fps_capsule


with open('parking_spots.pkl','rb') as f:
    spots = pickle.load(f)

with open("./models/model_rfc.p",'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(IMAGE_PATH)

while True :

    test , img = cap.read()
    if not test or img is None:
        break

frame = cv2.imread(IMAGE_PATH)

for x,y in spots:
    img = frame[y : y + BOX_HEIGHT, x : x + BOX_WIDTH]

    # Convert BGR (OpenCV) to RGB (skimage.io.imread format used in training)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to (50, 50) as done in training
    img = cv2.resize(img, (50, 50))

    # Normalize to [0, 1] range (skimage.transform.resize returns float values in [0, 1])
    img = img / 255.0

    img = img.flatten().reshape(1, -1)

    if model.predict(img)[0] == 1:
        color = (0, 0, 255)  # Red (BGR) for occupied/not_empty
    else:
        color = (0, 255, 0)  # Green (BGR) for empty

    cv2.rectangle(frame, (x,y),(x+BOX_WIDTH , y+BOX_HEIGHT),
                color,2)

cv2.imshow("Parking", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# saving it 

cv2.imwrite(("parking_results.jpeg"),frame)

print("---- Image Saved! -----")