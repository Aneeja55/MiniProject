import cv2
import os
import shutil
import mediapipe as mp

# Use relative paths based on script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'Data')
video_path = os.path.join(script_dir, 'Videos', 'TestVideo2.mp4')
BadFrame_path = os.path.join(script_dir, 'BadFrames')
detect_method = 'mediapipe'

# Clear/Create data directory
def dir_setup():
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
        currentframe += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total frames extracted: {currentframe}")
    vid.release()
    cv2.destroyAllWindows()

def process_frame(filename, source_dir, bad_dir):
    """Process a single frame to detect faces"""
    file_path = os.path.join(source_dir, filename)
    
    mp_face_detection = mp.solutions.face_detection
    
    try:
        img = cv2.imread(file_path)
        if img is None:
            return 'error'
        
        height, width, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_rgb)
            
            if not results.detections:
                # No face detected, move to BadFrames
                shutil.move(file_path, os.path.join(bad_dir, filename))
                return 'moved'
            else:
                # Face(s) detected, crop and save
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * width)
                    y = int(bboxC.ymin * height)
                    w = int(bboxC.width * width)
                    h = int(bboxC.height * height)
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    if w > 0 and h > 0:
                        face_crop = img[y:y+h, x:x+w]
                        
                        # Create Person Folder
                        person_folder = os.path.join(source_dir, f"person_{i}")
                        if not os.path.exists(person_folder):
                            os.makedirs(person_folder, exist_ok=True)
                        
                        # Resize to 224x224 (Standard for CNNs)
                        face_crop_resized = cv2.resize(face_crop, (224, 224))
                        
                        save_path = os.path.join(person_folder, filename)
                        cv2.imwrite(save_path, face_crop_resized)
                
                # Remove original frame after processing
                os.remove(file_path)
                return 'kept'
                
    except Exception as e:
        print(f"Error processing frame {filename}: {str(e)}")
        return 'error'

def detect_face_frames():
    """Sequential version - processes frames one by one"""
    moved_count = 0
    kept_count = 0
    error_count = 0

    print(f"Scanning frames for faces with {detect_method} method...")

    # Collect all image files
    all_files = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_files.append(filename)
    
    total_files = len(all_files)
    print(f"Found {total_files} frames to process")

    # Process each frame sequentially
    for i, filename in enumerate(all_files):
        result = process_frame(filename, data_dir, BadFrame_path)
        
        if result == 'moved':
            moved_count += 1
        elif result == 'kept':
            kept_count += 1
        elif result == 'error':
            error_count += 1
        
        # Progress update every 50 frames
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
            print(f"Processed {i + 1}/{total_files} frames...", end='\r')

    print("\n" + "-" * 30)
    print("Process Complete!")
    print(f"Total frames with no faces (moved to BadFrames): {moved_count}")
    print(f"Total frames with faces (kept and cropped): {kept_count}")
    if error_count > 0:
        print(f"Total frames with errors: {error_count}")

if __name__ == "__main__":
    dir_setup()
    extract_frames()
    detect_face_frames()
