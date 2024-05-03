import cv2
import matplotlib.pyplot as plt
import time

## Install the following dependencies
## $ pip install opencv-python
##
## GOTURN Files: https://github.com/Mogball/goturn-files
##
## Download the following files and place them in the same directory as this file
## goturn.prototxt
## goturn.caffemodel


# Helper functions to draw rectangle, display rectangle, and draw text
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


# Set up tracker
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# Change the index to change the tracker type
tracker_type = tracker_types[6]

# Initialize tracker
if tracker_type == "BOOSTING":
    tracker = cv2.legacy.TrackerBoosting.create()
elif tracker_type == "MIL":
    tracker = cv2.legacy.TrackerMIL.create()
elif tracker_type == "KCF":
    tracker = cv2.TrackerKCF.create()
elif tracker_type == "CSRT":
    tracker = cv2.legacy.TrackerCSRT.create()
elif tracker_type == "TLD":
    tracker = cv2.legacy.TrackerTLD.create()
elif tracker_type == "MEDIANFLOW":
    tracker = cv2.legacy.TrackerMedianFlow.create()
elif tracker_type == "GOTURN":
    tracker = cv2.TrackerGOTURN.create()
else:
    tracker = cv2.legacy.TrackerMOSSE.create()

# Read in the video
video = cv2.VideoCapture("race_car.mp4")
ok, frame = video.read()

if not video.isOpened():
    print("Error opening video file")
    exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make the window a quarter of the original size
    cv2.namedWindow("Object Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Tracker", width // 2, height // 2)


# Define a bounding box
bbox = (1300, 405, 160, 120)
displayRectangle(frame, bbox)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()

    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        drawRectangle(frame, bbox)
    else:
        drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # Display Info
    drawText(frame, tracker_type + " Tracker", (80, 60))

    cv2.imshow("Frame", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.05)

video.release()
cv2.destroyAllWindows()

# Release the video capture object and close the window
video.release()
cv2.destroyAllWindows()
