import pickle
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from SpotPicker import SAVE_FILE, BOX_WIDTH, BOX_HEIGHT

# -------------------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------------------
VIDEO_PATH = "carPark.mp4"  # Path to your test video file
SPOTS_PATH = SAVE_FILE      # Saved coordinates from SpotPicker

# Caffe Model Files
PROTOTXT = "./models/mAlexNet-on-CNRPark/deploy.prototxt"
MODEL = "./models/mAlexNet-on-CNRPark/snapshot_iter_942.caffemodel"

# -------------------------------------------------------------------------
# 2. Load Resources
# -------------------------------------------------------------------------
# Load parking spot coordinates
try:
    with open(SPOTS_PATH, "rb") as f:
        spots = pickle.load(f)
    print(f"[INFO] Loaded {len(spots)} parking spots.")
except FileNotFoundError:
    print(f"[ERROR] Could not find '{SPOTS_PATH}'. Run SpotPicker first!")
    exit()

# Load Caffe Model into OpenCV DNN
print("[INFO] Loading Caffe AlexNet Model...")
net = cv2.dnn.readNet(MODEL, PROTOTXT)
print("[INFO] Model loaded successfully.")

# Open Video Stream
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(
        f"[ERROR] Could not open video file '{VIDEO_PATH}'. Check file path!"
    )
    exit()

# -------------------------------------------------------------------------
# 3. Live Processing Loop
# -------------------------------------------------------------------------
while True:
    ret, frame = cap.read()

    # Loop video back to start when it reaches the end
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    free_spots = 0

    for x, y in spots:
        # Extract crop for current spot
        spot_crop = frame[y : y + BOX_HEIGHT, x : x + BOX_WIDTH]

        # Edge Safeguard: Skip out-of-bounds crops
        if (
            spot_crop is None
            or spot_crop.size == 0
            or spot_crop.shape[0] != BOX_HEIGHT
            or spot_crop.shape[1] != BOX_WIDTH
        ):
            continue

        # Preprocess patch into 4D Caffe Blob (227x227 input size, scaled 1/255.0)
        blob = cv2.dnn.blobFromImage(
            spot_crop,
            scalefactor=1.0 / 255.0,
            size=(227, 227),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
        )

        # Run model inference
        net.setInput(blob)
        output = net.forward()

        # Get class index: 0 = Occupied, 1 = Empty
        prediction = np.argmax(output[0])

        if prediction == 0:
            color = (0, 255, 0)  # Green (BGR) for Empty
        else:
            color = (0, 0, 255)  # Red (BGR) for Occupied
            free_spots += 1

        # Draw bounding box
        cv2.rectangle(
            frame, (x, y), (x + BOX_WIDTH, y + BOX_HEIGHT), color, thickness=2
        )

    # Render counter dashboard on top-left of video
    total_spots = len(spots)
    status_text = f"Free: {free_spots}/{total_spots}"

    # Draw semi-transparent background capsule for counter text
    cv2.rectangle(frame, (20, 20), (240, 70), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        status_text,
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Show result
    cv2.imshow("Smart Parking Space Detector (CNRPark AlexNet)", frame)

    # Press 'SPACE' or 'q' to exit execution loop
    key = cv2.waitKey(25) & 0xFF
    if key == ord(" ") or key == ord("q") or key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()