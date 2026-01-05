import face_recognition
import cv2
import os
import shutil
import concurrent.futures
import traceback

# Use relative paths based on script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'Data')
video_path = os.path.join(script_dir, 'Videos', 'TestVideo2.mp4')
BadFrame_path = os.path.join(script_dir, 'BadFrames')
Scale_factor = 0.25
detect_method='hog'

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
        print(f"Saved: {filename}")

        #cv2.imshow("Output", frame)
        currentframe += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total frames extracted: {currentframe}")
    vid.release()
    cv2.destroyAllWindows()

def process_frame_worker(file_info):
    filename,source_dir,bad_dir=file_info
    file_path=os.path.join(source_dir,filename)
    try:
        img=cv2.imread(file_path)
        if img is None:
            return 'error'
        small_frame=cv2.resize(img,(0,0),fx=Scale_factor,fy=Scale_factor)
        small_frame_rgb=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
        face_location=face_recognition.face_locations(small_frame_rgb,model='hog')
        if len(face_location)==0:
            shutil.move(file_path,os.path.join(bad_dir,filename))
            return 'moved'
        else:
            return 'kept'
    except Exception as e:
        print(f"Error processing frame {filename}: {str(e)}")
        traceback.print_exc()
        return 'error'
    
#Optimized version
def detect_face_frames():
    moved_count = 0
    kept_count = 0

    print(f"Scanning frames for faces with {detect_method} method...")

    all_files=[]
    for filename in os.listdir(data_dir):
        #file_path = os.path.join(data_dir, filename)
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_files.append((filename,data_dir,BadFrame_path))
        
    total_files=len(all_files)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=executor.map(process_frame_worker,all_files,chunksize=10)

        for i,results in enumerate(results):
            if results=='moved':
                moved_count+=1
            elif results=='kept':
                kept_count+=1
            if i%50==0:
                print(f"Processed {i}/{total_files} frames...",end='\r')

    print("\n"+"-"*30)
    print("Process Complete!")
    print(f"Total frames deleted: {moved_count}")
    print(f"Total frames kept: {kept_count}")

if __name__=="__main__":
    dir_setup()
    extract_frames()
    detect_face_frames()