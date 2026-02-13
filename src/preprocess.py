import cv2
import os
import shutil
import mediapipe as mp

# Standard Xception input size
IMG_SIZE = (224, 224)

def process_video_frames(video_path, output_root, detect_confidence=0.5):
    """
    Extracts faces from a video and saves them to output_root.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video file")
        return

    # Prepare directories
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_root, base_name)
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    print(f"Processing {base_name}...")
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=detect_confidence)
    
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = vid.read()
        if not success:
            break
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        results = face_detection.process(img_rgb)
        
        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)
                
                # Boundary checks
                x, y = max(0, x), max(0, y)
                w, h = min(width - x, w), min(height - y, h)
                
                if w > 0 and h > 0:
                    face_crop = frame[y:y+h, x:x+w]
                    # Resize to Xception standard
                    face_resized = cv2.resize(face_crop, IMG_SIZE)
                    
                    filename = f"frame_{frame_count}_face_{i}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), face_resized)
                    saved_count += 1
        
        frame_count += 1

    vid.release()
    print(f"Finished {base_name}: Processed {frame_count} frames, saved {saved_count} faces.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define Source Video Paths
    videos_dir = os.path.join(project_root, 'Videos')
    real_source = os.path.join(videos_dir, 'Real')
    fake_source = os.path.join(videos_dir, 'Fake')

    # Define Output Dataset Paths
    dataset_root = os.path.join(project_root, 'dataset', 'train')
    real_dest = os.path.join(dataset_root, 'Real')
    fake_dest = os.path.join(dataset_root, 'Fake')

    # Ensure output directories exist
    os.makedirs(real_dest, exist_ok=True)
    os.makedirs(fake_dest, exist_ok=True)

    # 1. Process Real Videos
    if os.path.exists(real_source):
        print(f"Processing Real videos from {real_source}...")
        for video_file in os.listdir(real_source):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                process_video_frames(
                    os.path.join(real_source, video_file), 
                    real_dest
                )
    else:
        print(f"Warning: {real_source} does not exist. Create it and add real videos.")

    # 2. Process Fake Videos
    # if os.path.exists(fake_source):
    #     print(f"Processing Fake videos from {fake_source}...")
    #     for video_file in os.listdir(fake_source):
    #         if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
    #             process_video_frames(
    #                 os.path.join(fake_source, video_file), 
    #                 fake_dest
    #             )
    # else:
    #     print(f"Warning: {fake_source} does not exist. Create it and add fake videos.")
        
    print("Preprocessing complete! Check the 'dataset/train' folder.")