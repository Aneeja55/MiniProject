import face_recognition
import cv2
import os
import shutil

# Use relative paths based on script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'Data')
video_path = os.path.join(script_dir, 'Videos', 'TestVideo2.mp4')
BadFrame_path = os.path.join(script_dir, 'BadFrames')

# Clear/Create data directory 
if os.path.exists(data_dir):
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
else:
    os.makedirs(data_dir)

if not os.path.exists(BadFrame_path):
        os.makedirs(BadFrame_path)
else:
    for entry in os.listdir(BadFrame_path):
        path = os.path.join(BadFrame_path, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def extract_frames():
    vid = cv2.VideoCapture(video_path)
    currentframe = 0

    if not vid.isOpened():
        print("Error: Could not open video file")
        exit()

    print("Extracting frames...")

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

def detect_face_frames():
    detect_method='hog'
    delete_count = 0
    kept_count = 0

    print(f"Scanning frames for faces with {detect_method} method...")

    # Each frame is checked if it contains a face
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if not file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        image=face_recognition.load_image_file(file_path)
        face_loacations=face_recognition.face_locations(image,number_of_times_to_upsample=2, model=detect_method)
        if len(face_loacations) == 0:
            shutil.move(file_path, BadFrame_path)
            delete_count += 1
        else:
            kept_count += 1

    print(f"Total frames deleted: {delete_count}")
    print(f"Total frames kept: {kept_count}")


extract_frames()
detect_face_frames()