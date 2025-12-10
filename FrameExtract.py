import cv2
import os
import shutil

# Use relative paths based on script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
video_path = os.path.join(script_dir, 'ai', 'vid2.mp4')

if os.path.exists(data_dir):
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
else:
    os.makedirs(data_dir)

vid = cv2.VideoCapture(video_path)
currentframe = 0

if not vid.isOpened():
    print("Error: Could not open video file")
    exit()

while True:
    success, frame = vid.read()
    if not success:
        break

    filename = os.path.join(data_dir, f'frame{currentframe}.jpg')
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

    cv2.imshow("Output", frame)
    currentframe += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Total frames extracted: {currentframe}")
vid.release()
cv2.destroyAllWindows()