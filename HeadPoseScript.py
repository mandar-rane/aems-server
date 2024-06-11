import cv2
import mediapipe as mp
import numpy as np
import time
import socketio

sio = socketio.Client()

sio.connect('http://localhost:3000')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.1, min_tracking_confidence=0.5,max_num_faces=4, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

desired_width = 640  # Set your desired width
desired_height = 480  # Set your desired height

cap = cv2.VideoCapture("rtsp://192.168.95.198:5543/live/channel0")
# cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (desired_width, desired_height))

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    
    # Variables for counting people's attentiveness in different directions
    attentive_count = 0
    total_people = 0
    direction_counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'straight': 0}

    if results.multi_face_landmarks:
        total_people = len(results.multi_face_landmarks)

        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            face_3d = []
            face_2d = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)        
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Update direction counts based on pose estimation results
            if y < -15:
                direction_counts['left'] += 1
            elif y > 15:
                direction_counts['right'] += 1
            elif x > 15:
                direction_counts['up'] += 1
            elif x < -15:
                direction_counts['down'] += 1
            else:
                direction_counts['straight'] += 1

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255,0,0), 3)

            
            
            mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec= drawing_spec)

        # Determine the majority direction
        majority_direction = max(direction_counts, key=direction_counts.get)

        # Calculate the attentive percentage
        attentive_count = direction_counts[majority_direction]
        attentive_percentage = (attentive_count / total_people) * 100 if total_people != 0 else 0
        sio.emit("update_variable", attentive_percentage) 
        sio.emit("attendance", total_people)
        
        time.sleep(0.01)
        # Display the attentive percentage and the majority direction
        cv2.putText(image, f'Attentive: {attentive_percentage:.2f}%', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Majority Direction: {majority_direction}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    end = time.time()
    total_time = end - start

    if total_time > 0:
        fps = 1 / total_time
    else:
        fps = 0

    print("FPS: ", fps)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose ESTIMATION', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
            

        

