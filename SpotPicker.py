import cv2
import pickle
from utils import mouse_click

# 1. Configuration
IMAGE_PATH = "carPark.png" # 'parking.jpg'
SAVE_FILE = "CarParkPos.pkl" # 'parking_spots.pkl'

BOX_WIDTH = 107 #90
BOX_HEIGHT = 48 #170

# 2. Try to load existing spots, or start fresh
try:
    with open(SAVE_FILE, 'rb') as f:
        spots_list = pickle.load(f)
except FileNotFoundError:
    spots_list = []


if __name__ == '__main__':
    # 4. Main Display Loop
    cv2.namedWindow('Parking Spot Picker')
    cv2.setMouseCallback('Parking Spot Picker', mouse_click, param=(spots_list, BOX_WIDTH, BOX_HEIGHT, SAVE_FILE))

    while True:
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"Error: Could not load image from '{IMAGE_PATH}'. Check the path!")
            break

        # Draw all marked spots
        for pos in spots_list:
            x, y = pos
            # Draw box: Magenta border with box counter
            cv2.rectangle(img, (x, y), (x + BOX_WIDTH, y + BOX_HEIGHT), (255, 0, 255), 1)

        # Show image and listen for keys
        cv2.imshow('Parking Spot Picker', img)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' or 'ESC' to exit
        if key == ord('q') or key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()