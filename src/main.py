import torch
import numpy as np
import cv2
import time

from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Running on device: {}'.format(device))


def preprocess_frame(frame):
  """Preprocesses a frame for face detection and embedding calculation."""
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return frame_rgb



def get_embeddings(frame_rgb, mtcnn, resnet):
    """Detects faces, aligns them, and calculates embeddings."""
    boxes, _ = mtcnn.detect(frame_rgb, landmarks=False)
    if boxes is None:
        return None, None

    aligned_faces = []
    for box in boxes:
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        cropped_face = frame_rgb[y1:y2, x1:x2]
        if cropped_face.size == 0:
            continue
        aligned_face = cv2.resize(cropped_face, (160, 160))  # Resize to a fixed size
        aligned_faces.append(aligned_face)

    aligned_faces = np.array(aligned_faces)
    if len(aligned_faces) == 0:
        return None, None  # No valid faces found

    aligned_faces = torch.tensor(aligned_faces).permute(0, 3, 1, 2).float().to(device)
    embeddings = resnet(aligned_faces).detach().cpu()
    return boxes, embeddings


# Define MTCNN and InceptionResnetV1 modules
mtcnn = MTCNN(
     image_size=160, margin=0, min_face_size=20,
     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
     device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Open the video file
video_file = '/home/fadymaher/git-fedora/face_recognition_haibco/data/People-Walking.mp4'
video = cv2.VideoCapture(video_file)

# Define video writer for saving output
output_video_file = '/home/fadymaher/git-fedora/face_recognition_haibco/output/output_detected_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose a suitable codec for your needs
output_video = cv2.VideoWriter(output_video_file, fourcc, 25.0, (640, 360))

if not video.isOpened():
    print("Error opening video stream or file not found!")
    exit()

start_time = time.time()
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("No more frames to read from video")
        break

    frame_count += 1

    # # Skip processing every 5th frame for performance
    if frame_count % 5 != 0:
        continue

    with torch.no_grad():
        frame_rgb = preprocess_frame(frame)
        boxes,embeddings = get_embeddings(frame_rgb, mtcnn, resnet)

        if embeddings is None:
            print('No faces detected by MTCNN')
            continue

        # Process the embeddings (recognition logic)
        # ...

        # Display the bounding boxes
        for box in boxes:
    # Convert bounding box coordinates to integers
          x1, y1, x2, y2 = map(int, box)
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


          cv2.imshow('frame', frame)
          output_video.write(frame)  # Write frame to output video

          if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Calculate and display FPS every second
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

video.release()
output_video.release()
cv2.destroyAllWindows()
