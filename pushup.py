import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from gtts import gTTS
import base64
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from threading import Thread
import time

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return b64

def create_audio_html(b64_data):
    html_string = f"""
        <audio autoplay="true">
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

class VideoProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = "down"
        self.sequence_stage = "Initial"
        self.feedback_status = ""
        self.last_feedback = ""
        self.prev_time = 0
        self.frame_rate = 5  # Set the frame rate (e.g., 5 frames per second)
        self.prev_stage = "down"

    def recv(self, frame):

        self.feedback_status = ""

        frm = frame.to_ndarray(format="bgr24")
        small_frame = cv2.resize(frm, (640, 480))  # Downscale the frame
        image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark

            # Define landmarks for push-up exercise
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle between the shoulders, elbows, and wrists
            angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if angle1 >= 160 and angle2 >= 160 and self.sequence_stage != "up" and self.prev_stage == "down":
                self.sequence_stage = "up"
                self.counter += 1
                print(self.counter)
                self.stage = "Push Up"
                # self.feedback_status = "Arms straightened. Go to curl."
            elif 90 <= angle1 <= 140 and 90 <= angle2 <= 140 and self.sequence_stage != "down":
                self.sequence_stage = "down"
                self.stage = "Lie Down"
                

            # Inside the while loop where you detect the push-up stage
            if self.sequence_stage == "up":
                self.feedback_status = "Good Push Up!"
            elif self.sequence_stage == "down":
                self.feedback_status = "Lower down!"

            # Ensure all required landmarks are detected
            if all(landmarks[landmark].visibility > 0.5 for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                                        mp_pose.PoseLandmark.LEFT_ELBOW.value,
                                                                        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                                                        mp_pose.PoseLandmark.LEFT_WRIST.value,
                                                                        mp_pose.PoseLandmark.RIGHT_WRIST.value]):
                # Get the left elbow landmark coordinates
                left_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1])
                left_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])
                
                # Get the right elbow landmark coordinates
                right_elbow_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1])
                right_elbow_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0])


                # Display the angle value near the left elbow position
                cv2.putText(image, str(angle1), 
                            (left_elbow_x, left_elbow_y),  # Use left elbow coordinates
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            
                # Display the angle value near the right elbow position
                cv2.putText(image, str(angle2), 
                            (right_elbow_x, right_elbow_y),  # Use right elbow coordinates
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Stage: {self.stage}",
                        (10, image.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
            cv2.putText(image, f"Feedback: {self.feedback_status}",
                        (10, image.shape[0] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
        except Exception as e:
            print(e)

        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 1, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.sequence_stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        return av.VideoFrame.from_ndarray(image, format='bgr24')
    
    def get_feedback_status(self):
        return self.feedback_status

def app():
    st.set_page_config(layout="wide")
    st.title("Push Up Exercise")
    remind_text = "**Make sure to allow access of your camera and speaker. Refresh the website if there is a lag.**"
    st.markdown(remind_text)
    remark_text = "**Perfect angle for a push up is more than 160 degree of your straightened arms. Make sure to show your front upper body during a push up to a webcam.**"
    st.markdown(remark_text)

    webrtc_ctx = webrtc_streamer(
    key="full-body-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": {"frameRate": {"ideal": 15}}, "audio": False},
    video_html_attrs={
        "style": {"width": "50%", "margin": "0 auto", "border": "5px purple solid"},
        "controls": False,
        "autoPlay": True,
    },
    async_processing=True,
)

    video_processor = VideoProcessor()

    last_feedback = ""
    filename = ""

    if webrtc_ctx.video_processor:
        video_processor = webrtc_ctx.video_processor
        while True:
            feedback_status = video_processor.get_feedback_status()
            # print(f"Latest Update: {feedback_status}")
            time.sleep(0.2)  # Delay of 0.1 seconds (100 milliseconds)
            if feedback_status == "Good Push Up!":
                filename = 'good_push_up.mp3'
                if feedback_status != last_feedback:
                    play_audio(feedback_status, filename)
                    last_feedback = feedback_status
            elif feedback_status == "Lower down!":
                filename = 'lower_down.mp3'
                if feedback_status != last_feedback:
                    play_audio(feedback_status, filename)
                    last_feedback = feedback_status


if __name__ == '__main__':
    app()
