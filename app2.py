from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Initialize dlib models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmark model points
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

dist_coeffs = np.zeros((4, 1))

# Camera parameters
def get_camera_params(cap):
    focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    center = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    return camera_matrix
def detect_focus():
    cap = cv2.VideoCapture(0)
    camera_matrix = get_camera_params(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            landmarks_2d = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),   # Chin
                (landmarks.part(36).x, landmarks.part(36).y), # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y), # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y), # Left mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            # Calculate head pose
            _, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, landmarks_2d, camera_matrix, dist_coeffs
            )
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))[6]
            pitch, yaw, roll = angles.flatten()

            # Draw head pose angles
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Get eye regions
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]

            def get_eye_center(eye_points):
                return int(sum(p.x for p in eye_points) / len(eye_points)), \
                       int(sum(p.y for p in eye_points) / len(eye_points))

            left_eye_center = get_eye_center(left_eye)
            right_eye_center = get_eye_center(right_eye)

            # Draw eye centers
            cv2.circle(frame, left_eye_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 2, (0, 255, 0), -1)

            # Focus determination
            focused = (-10 < yaw < 10) and (-10 < roll < 10)

            # Display focus status
            status = "FOCUSED" if focused else "NOT FOCUSED"
            color = (0, 255, 0) if focused else (0, 0, 255)
            cv2.putText(frame, f"Focus: {status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_focus(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=9000)
