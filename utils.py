import cv2
import pickle



def mouse_click(event, x, y, flags, param):
    spots_list, bw, bh, file = param
    
    # Left Click -> ADD a new parking spot
    if event == cv2.EVENT_LBUTTONDOWN:
        spots_list.append((x, y))
        print(f"Added spot at ({x}, {y})")

    # Right Click -> DELETE an existing parking spot
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(spots_list):
            x1, y1 = pos
            # Check if right-click occurred inside an existing box
            if x1 <= x <= x1 + bw and y1 <= y <= y1 + bh:
                spots_list.pop(i)
                print(f"Removed spot at ({x1}, {y1})")
                break

    # Save updated list to disk immediately
    with open(file, 'wb') as f:
        pickle.dump(spots_list, f)