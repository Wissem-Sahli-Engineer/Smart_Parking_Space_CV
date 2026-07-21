import cv2
import pickle
import time



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

def get_fps(cap, pTime,type='default'):
    if type == "default":
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        return fps, pTime

    elif type =="cap":
        fps= cap.get(cv2.CAP_PROP_FPS)
        if fps<= 0:
            return 30, pTime
        return fps, pTime
    else:
        return 30, pTime

def draw_fps_capsule(img, fps):
    """
    Draws a clean modern capsule for the frame-rate in the top-left corner.
    """
    fps_overlay = img.copy()
    cv2.rectangle(fps_overlay, (1175, 15), (1315, 55), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(fps_overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, f"FPS: {int(fps)}", (1200, 42), cv2.FONT_HERSHEY_SIMPLEX, 
        0.55, (0, 255, 255), 1, cv2.LINE_AA)