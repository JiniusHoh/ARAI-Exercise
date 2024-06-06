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


from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from gtts import gTTS
import base64

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

        for x,y,w,h in faces:
            cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return b64

def create_audio_html(b64_data):
    html_string = f"""
        <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64_data}" type="audio/mp3">
        </audio>
    """
    return html_string

# Initialize MediaPipe's drawing utilities and pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)

def app():
    st.set_page_config(layout="wide")  # Set the layout to wide
    st.title("Left Bicep Curl Exercise")
    remind_text = "Make sure to allow access to your camera and speaker. Refresh the website if there is a lag."
    st.markdown(remind_text)  # Display remark below the frame
    remark_text = "If there is no autoplay sound, press the most bottom audio play button once to autoplay the feedback sound. Might need some time to start autoplay the feedback sound."
    st.markdown(remark_text)  # Display remark below the frame

    # GIF placeholder
    gif_path = 'bicep_curl.gif'  # Replace with the path to your GIF file
    # CSS to control the size of the GIF
    st.markdown(
        """
        <style>
        .small-gif {
            width: 1000px;
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    gif_placeholder = st.empty()

    start_button = st.button('Start Exercise')

    if start_button:
        gif_placeholder.empty()  # Clear the GIF
        stframe = st.empty()

        def autoplay_audio(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            return b64

        def create_audio_html(b64_data):
            html_string = f"""
                <audio controls autoplay="true">
                    <source src="data:audio/mp3;base64,{b64_data}" type="audio/mp3">
                </audio>
            """
            return html_string

        def play_audio(feedback_text, filename):
            tts = gTTS(feedback_text)
            tts.save(filename)
            audio_file = filename

            b64_data = autoplay_audio(audio_file)
            html_content = create_audio_html(b64_data)
            st.markdown(html_content, unsafe_allow_html=True)

        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return int(angle)

        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                                    rtc_configuration=RTCConfiguration(
                                        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                                    )
                        )

        # Set initial values for stage and counter
        counter = 0
        stage = "down"
        sequence_stage = "Initial"
        feedback_status = ""
        last_feedback = ""

        while webrtc_ctx.video_processor:
            frame = webrtc_ctx.video_receiver.get_frame()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                results = pose.process(frame)

                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle > 160 and sequence_stage != "Bicep Curl":
                        sequence_stage = "Straighten Arms"
                        stage = "down"
                        feedback_status = "Arms straightened. Go to curl."
                    elif sequence_stage == "Straighten Arms" and angle < 110 and stage == "down":
                        sequence_stage = "Bicep Curl"
                        stage = "up"
                        counter += 1
                        print(counter)

                    # Update feedback status on each frame
                    if sequence_stage == "Bicep Curl":
                        if angle < 45:
                            feedback_status = "Lower down your arm."
                            filename = 'lower_arm.mp3'
                        elif 45 <= angle <= 65:
                            feedback_status = "Good Curl!"
                            filename = 'goodcurl.mp3'
                        # elif 65 < angle <= 105:
                        #     feedback_status = "Raise up your arm."
                        #     filename = 'raise_arm.mp3'
                        elif angle > 160 and stage == "up":
                            sequence_stage = "Straighten Arms"
                            stage = "down"
                            feedback_status = "Arms straightened. Go to curl."
                            last_feedback = feedback_status
                            continue  # Skip the rest of the loop

                        # Only give feedback if it's different from the last feedback given
                        if feedback_status != last_feedback:
		   	    play_audio(feedback_status, filename)
                            last_feedback = feedback_status

                    # Ensure the left elbow landmark is detected
                    if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5:
                        # Get the left elbow landmark coordinates
                        left_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1])
                        left_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0])

                        # Display the angle value near the left elbow position
                        cv2.putText(frame, str(angle),
                                    (left_elbow_x, left_elbow_y),  # Use left elbow coordinates
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Adjust sequence stage text
                    cv2.putText(frame, f"Stage: {sequence_stage}",
                                (10, frame.shape[0] - 55),  # Position at the left bottom corner
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # Adjust feedback status text
                    cv2.putText(frame, f"Feedback: {feedback_status}",
                                (10, frame.shape[0] - 20),  # Position above the sequence stage text
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)
                    cv2.putText(frame, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 1, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(counter), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'STAGE', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

                    # Resize the image to make it larger
                    frame = cv2.resize(frame, (1200, 800))  # Resize the image

                    stframe.image(frame, channels="BGR")  # Display RGB frame

                except Exception as e:
                    print(e)
                    pass

    else:
        gif_html = f"""
        <div style="text-align: center;">
            <img src="data:image/gif;base64,{base64.b64encode(open(gif_path, "rb").read()).decode()}" class="small-gif">
        </div>
        """
        st.markdown(gif_html, unsafe_allow_html=True)
        st.markdown('**Perfect angle for a bicep curl is 45 degree to 60 degree.**')
        st.markdown('**Try now with your left arm! Make sure to show your upper body with your left arm into your webcam.**')

if __name__ == '__main__':
    app()
