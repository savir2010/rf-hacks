from flask import Flask, render_template, Response, request, jsonify
import base64
import cv2
import dlib
import numpy as np
from geopy.geocoders import Nominatim
import quiz_generate as qg
import assistant

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
state = 'Amazing Job Staying Focused'
score = 100
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
    global score
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
            axis = np.float32([
                [500, 0, 0],  # X-axis (red)
                [0, 500, 0],  # Y-axis (green)
                [0, 0, 500]   # Z-axis (blue)
            ])
            img_pts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            nose_tip = tuple(landmarks_2d[0].astype(int))
            for idx, color in zip(range(3), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                cv2.line(frame, nose_tip, tuple(img_pts[idx].ravel().astype(int)), color, 3)
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

            # Draw eye
            for point in left_eye + right_eye:
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
            cv2.circle(frame, left_eye_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 2, (0, 255, 0), -1)

            # Focus determination
            focused = (-10 < yaw < 10) and (-10 < roll < 10)

            # Display focus status
            status = "FOCUSED" if focused else "NOT FOCUSED"
            if status == "NOT FOCUSED":
                score -= 0.1
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

@app.route('/focus', methods=['GET','POST'])
def focus():
    topic = request.form.get('topic')

    print(topic)
    if topic:
        with open("topics.txt", "w") as file:
            file.write(topic)
        quiz_lst = qg.quiz_response(topic)  
        print(quiz_lst)
        return render_template('focus.html',
                            score=state,
                            topic=topic, 
                            question=quiz_lst[0], 
                            correct_answer=quiz_lst[1], 
                            choice1=quiz_lst[2],
                            choice2=quiz_lst[3],
                            choice3=quiz_lst[4])
    return render_template('focus.html')
    

@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.get_json()  # Get the incoming JSON data
    user_message = data.get('message', '')  # Extract the message from the request

    if user_message:
        # For demonstration, let's just echo the message back.
        response = assistant.therapist_response(user_message)
    else:
        response = "Sorry, I couldn't understand your message."
    
    # Return the response as a JSON object
    return jsonify({"response": response})

@app.route('/video_feed')
def video_feed():
  
    return Response(detect_focus(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    try:
        # Extract the Base64 image from the POST request
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])  # Decode Base64 string
        np_arr = np.frombuffer(image_data, np.uint8)  # Convert to numpy array
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode image into OpenCV format

        # Process the frame (e.g., focus detection)
        # For example, save the frame to check if it's working
        cv2.imwrite('received_frame.jpg', frame)

        return "Frame received and processed", 200
    except Exception as e:
        print(f"Error processing frame: {e}")
        return "Error", 500

@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
    return render_template("chatbot.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)

