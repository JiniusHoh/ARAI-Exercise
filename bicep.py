# import cv2
# import mediapipe as mp
# import numpy as np
# import streamlit as st
# from gtts import gTTS
# import base64
# from webcam import webcam


# def autoplay_audio(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode()
#     return b64

# def create_audio_html(b64_data):
#     html_string = f"""
#         <audio controls autoplay="true">
#             <source src="data:audio/mp3;base64,{b64_data}" type="audio/mp3">
#         </audio>
#     """
#     return html_string


# # Initialize MediaPipe's drawing utilities and pose module
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return int(angle)

# def app():
#     st.set_page_config(layout="wide")  # Set the layout to wide
#     st.title("Left Bicep Curl Exercise")
#     remind_text = "Make sure to allow access to your camera and speaker. Refresh the website if there is a lag."
#     st.markdown(remind_text)  # Display remark below the frame
#     remark_text = "If there is no autoplay sound, press the most bottom audio play button once to autoplay the feedback sound. Might need some time to start autoplay the feedback sound."
#     st.markdown(remark_text)  # Display remark below the frame

#     # GIF placeholder
#     gif_path = 'bicep_curl.gif'  # Replace with the path to your GIF file
#     # CSS to control the size of the GIF
#     st.markdown(
#         """
#         <style>
#         .small-gif {
#             width: 1000px;
#             height: auto;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#     gif_placeholder = st.empty()

#     start_button = st.button('Start Exercise')


#     if start_button:
#         gif_placeholder.empty()  # Clear the GIF
#         stframe = st.empty()

#         # cap = cv2.VideoCapture(0)
#         captured_image = webcam()

#         # Set initial values for stage and counter
#         counter = 0
#         stage = "down"
#         sequence_stage = "Initial"
#         feedback_status = ""
#         last_feedback = ""

#         def play_audio(feedback_text, filename):
#             tts = gTTS(feedback_text)
#             tts.save(filename)
#             audio_file = filename

#             b64_data = autoplay_audio(audio_file)
#             html_content = create_audio_html(b64_data)
#             st.markdown(html_content, unsafe_allow_html=True)

#         with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#             while captured_image is True:
#                 ret, frame = captured_image.read()
#                 if not ret:
#                     break

#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing
#                 image.flags.writeable = False

#                 results = pose.process(image)

#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

#                 try:
#                     landmarks = results.pose_landmarks.landmark

#                     shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                     elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                     wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                     angle = calculate_angle(shoulder, elbow, wrist)

#                     if angle > 160 and sequence_stage != "Bicep Curl":
#                         sequence_stage = "Straighten Arms"
#                         stage = "down"
#                         feedback_status = "Arms straightened. Go to curl."
#                     elif sequence_stage == "Straighten Arms" and angle < 110 and stage == "down":
#                         sequence_stage = "Bicep Curl"
#                         stage = "up"
#                         counter += 1
#                         print(counter)

#                     # Update feedback status on each frame
#                     if sequence_stage == "Bicep Curl":
#                         if angle < 45:
#                             feedback_status = "Lower down your arm."
#                             filename = 'lower_arm.mp3'
#                         elif 45 <= angle <= 65:
#                             feedback_status = "Good Curl!"
#                             filename = 'goodcurl.mp3'
#                         # elif 65 < angle <= 105:
#                         #     feedback_status = "Raise up your arm."
#                         #     filename = 'raise_arm.mp3'
#                         elif angle > 160 and stage == "up":
#                             sequence_stage = "Straighten Arms"
#                             stage = "down"
#                             feedback_status = "Arms straightened. Go to curl."
#                             last_feedback = feedback_status
#                             continue  # Skip the rest of the loop

#                         # Only give feedback if it's different from the last feedback given
#                         if feedback_status != last_feedback:
#                             play_audio(feedback_status, filename)
#                             last_feedback = feedback_status

#                     # Ensure the left elbow landmark is detected
#                     if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5:
#                         # Get the left elbow landmark coordinates
#                         left_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1])
#                         left_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])

#                         # Display the angle value near the left elbow position
#                         cv2.putText(image, str(angle),
#                                     (left_elbow_x, left_elbow_y),  # Use left elbow coordinates
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                     # Adjust sequence stage text
#                     cv2.putText(image, f"Stage: {sequence_stage}",
#                                 (10, image.shape[0] - 55),  # Position at the left bottom corner
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#                     # Adjust feedback status text
#                     cv2.putText(image, f"Feedback: {feedback_status}",
#                                 (10, image.shape[0] - 20),  # Position above the sequence stage text
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#                 except Exception as e:
#                     print(e)
#                     pass

#                 cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
#                 cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 1, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, str(counter), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 cv2.putText(image, 'STAGE', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Draw landmarks and connections
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

#                 # Resize the image to make it larger
#                 image = cv2.resize(image, (1200, 800))  # Resize the image

#                 stframe.image(image, channels="BGR")  # Display RGB frame

#         # captured_image.release()

#     else:
#         gif_html = f"""
#         <div style="text-align: center;">
#             <img src="data:image/gif;base64,{base64.b64encode(open(gif_path, "rb").read()).decode()}" class="small-gif">
#         </div>
#         """
#         st.markdown(gif_html, unsafe_allow_html=True)
#         st.markdown('**Perfect angle for a bicep curl is 45 degree to 60 degree.**')
#         st.markdown('**Try now with your left arm! Make sure to show your upper body with your left arm into your webcam.**')

# if __name__ == '__main__':
#     app()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import streamlit as st
from PIL import Image
from datetime import datetime, timezone, timedelta


# 録画開始時刻
START_REC_TIME = {"H": 8, "M": 30, "S": 0, "ms": 0}
# 録画終了時刻
END_REC_TIME = {"H": 9, "M": 00, "S": 0, "ms": 0}
# 動画保存先
VIDEO_DIR = "video"


st.title('🎥Remote cam app')
datetime_loc = st.empty()

# カメラデバイス設定
device = user_input = st.text_input("input your video/camera device", "0")
if device.isnumeric():
    device = int(device)

image_loc = st.empty()

d_today = datetime.now()
strtime = d_today.strftime('%Y%m%d_%H:%M:%S')
strday = d_today.strftime('%Y%m%d')
save_path = "./{}/{}".format(VIDEO_DIR, strday)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 動画ファイルの保存設定
cap = cv2.VideoCapture(device)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(
    'm', 'p', '4', 'v')
video = cv2.VideoWriter(
    '{}/{}.mp4'.format(save_path, strtime), fourcc, fps, (w, h))


def main():
    while cap.isOpened:
        # 時刻表示設定
        now = datetime.now(timezone(timedelta(hours=9))
                           ).strftime("%Y/%m/%d %H:%M:%S")
        datetime_loc.text(now)
        # 録画時刻設定
        current_time = datetime.now()
        at6am = current_time.replace(hour=START_REC_TIME["H"],
                                     minute=START_REC_TIME["M"],
                                     second=START_REC_TIME["S"],
                                     microsecond=START_REC_TIME["ms"]
                                     )
        at6pm = current_time.replace(hour=END_REC_TIME["H"],
                                     minute=END_REC_TIME["M"],
                                     second=END_REC_TIME["S"],
                                     microsecond=END_REC_TIME["ms"]
                                     )
        # 映像取得
        ret, img = cap.read()
        if ret:
            # テキスト描写
            cv2.putText(img,
                        text=str(now),
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            # 録画開始/終了条件確認
            if ((current_time >= at6am) and (current_time <= at6pm)):
                # 録画処理
                video.write(img)
                cv2.putText(img,
                            text="REC",
                            org=(570, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_AA)

            # 映像表示設定
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_loc.image(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == '__main__':
    main()
