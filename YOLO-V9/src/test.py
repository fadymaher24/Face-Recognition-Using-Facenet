import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
from dlib import face_recognition_model_v1, cnn_face_detection_model_v1

import subprocess
import face_recognition

def resize_frame(frame, scale_percent):
    """Function to resize an image by a percentage scale."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


def detect_and_recognize_faces(detector, gray, face):
    """
    Detects and recognizes faces in a given image region using MTCNN.

    Args:
        detector: The face detector object (dlib.cnn_face_detection_model_v1).
        gray: The grayscale image containing a potential face.
        face: The face region detected by the detector.

    Returns:
        A list containing the face bounding box and recognition result (if applicable).
    """

    try:
        # Extract face bounding box coordinates
        (x, y, w, h) = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
        # Return face information
        return {"bbox": (x, y, w, h)}

    except Exception as e:
        print(f"Error during face detection: {e}")
        return []
    

def compare_face_encodings(known_face_encoding, face_encoding):
    # Compute the Euclidean distance between the two face encodings
    return np.linalg.norm(np.array(known_face_encoding) - np.array(face_encoding))

# distance = compare_face_encodings(known_face_encoding, face_encoding)


def main():
    # Load YOLO model (assuming you have a pre-trained YOLOv9c model)
    model = YOLO("models/yolov9c.pt")
    
    names = model.model.names

    # Load MTCNN face detector (optional for face recognition)
    detector = cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
    # Load OpenFace face recognizer (if you need face recognition)
    
    face_recognizer = face_recognition_model_v1("models/nn4.small2.v1.t7")

    # Video capture setup (replace with your video source)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video capture device.")
        return

    # Output video setup (optional)
    w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 cap.get(cv2.CAP_PROP_FPS))
    output_video_file = '/home/fadymaher/git-fedora/face_recognition-YOLO/YOLO-V9/output/object_tracking_with_recognition.mp4'
    result = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', output_video_file
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Track history dictionary
    track_history = defaultdict(lambda: [])

    # Known face encodings (replace with your data)
    known_face_encodings = []
    known_face_names = []

    # def load_known_faces():
        # ... (replace with logic to load encodings and names from dataset) ...
        # ... populate known_face_encodings and known_face_names lists ...

    # load_known_faces()  # Call to load known faces before the loop

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error reading frame from video capture.")
            break
        frame_resized = resize_frame(frame, 50)  # Resize frame to 50% for faster processing

        # Perform YOLO object detection and tracking
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.tolist()
            # track_ids = results[0].boxes.id.int().tolist()
            confs = results[0].boxes.conf.float().tolist()

            # Annotator for drawing labels and boxes
            annotator = Annotator(frame, line_width=2)
            track_id = 0
            for box, cls in zip(boxes, clss):
                # Check if the detected class is "person" (or your desired class)
                if names[int(cls)] == "person":
                    # track_id for only persons
                    track_id += 1
                    # Draw bounding box and label
                    annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} (ID: {track_id})")

                    # Extract the person's bounding box
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    person_roi = frame[y1:y2, x1:x2]

                    # Convert to grayscale (MTCNN works better with grayscale)
                    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                    # Perform face detection using MTCNN
                    faces = detector(gray, 1)

                    # Perform face recognition (if necessary)
                    face_results = []
                    for face in faces:
                        face_data = detect_and_recognize_faces(detector, gray, face)
                        if face_data:  # If a face is detected within the person ROI
                            face_x, face_y, face_w, face_h = face_data["bbox"]
                            # Extract the face region for recognition
                            face_img = person_roi[face_y:face_y + face_h, face_x:face_x + face_w]

                            # Perform face recognition using the loaded model (replace with your logic)
                            face_encoding = face_recognizer.compute_face_descriptor(face_img, face_data["bbox"])
                            match_results = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            name = "Unknown"

                            # Identify the recognized face (if applicable)
                            if True in match_results:
                                first_match_index = match_results.index(True)
                                name = known_face_names[first_match_index]

                            # Perform safety checks based on your criteria (replace with your logic)
                            # is_safe = perform_safety_checks(name)  # Implement your safety check function

                            face_results.append({
                                "bbox": (face_x, face_y, face_w, face_h),
                                "name": name,
                                # "is_safe": is_safe
                            })

                    # Draw face bounding boxes and recognition results (if applicable)
                    for face_result in face_results:
                        face_x, face_y, face_w, face_h = face_result["bbox"]
                        cv2.rectangle(person_roi, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0) if face_result["is_safe"] else (0, 0, 255), 2)
                        # cv2.putText(person_roi, f"{face_result['name']} ({'Safe' if face_result['is_safe'] else 'Unsafe'})", (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if face_result["is_safe"] else (0, 0, 255), 2)

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

        # Draw tracking lines and labels (optional)
        # ... Implement logic to draw tracking lines and labels based on track_history ...

        # Write frame to output video
        result.write(frame)
        
        # Convert the resized frame to bytes and write it to ffmpeg's stdin
        ffmpeg_process.stdin.write(frame_resized.tobytes())

        # Handle user input for quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    result.release()
    cap.release()
     
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
