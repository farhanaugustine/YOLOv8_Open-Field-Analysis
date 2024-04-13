import cv2
from ultralytics import YOLO

# Load your custom model
model = YOLO(r'D:\YoloV8_Custom_Model\OFT_Model.pt') # path to your model

# Load video
video_path = r'D:\YoloV8_Custom_Model\20230904_Mouse_196_OFT_720p_30fps.mp4' # name of your video
cap = cv2.VideoCapture(video_path)

while True:
    # Read frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLO model on frame
    results = model.track(source=frame, tracker='bytetrack.yaml', persist=True, conf=0.0, iou=0.1) # adjust these parameters as needed
    # Display the frame
    frame_=results[0].plot()
    cv2.imshow('frame', frame_)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'): # change the keybindings to any keyboard key
        break

# Release the VideoCapture and close the windows
cap.release()
cv2.destroyAllWindows()
