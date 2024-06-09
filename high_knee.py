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

    def recv(self, frame):

        self.feedback_status = ""

        frm = frame.to_ndarray(format="bgr24")
        small_frame = cv2.resize(frm, (640, 480))  # Downscale the frame
        image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            

            angle = calculate_angle(right_hip, right_knee, right_ankle)

            if angle < 120 and self.sequence_stage != "High Knee":  # Condition for starting "High Knee" stage
                self.sequence_stage = "High Knee"
                self.stage = "up"
            elif self.sequence_stage == "High Knee" and angle >= 160 and self.stage == "up":
                self.sequence_stage = "Straighten Legs"
                self.stage = "down"
                self.counter += 1
                print(self.counter)
            # Update feedback status on each frame
            if self.sequence_stage == "High Knee":
                if 85 <= angle <= 125:
                    self.feedback_status = "Good High Knee!"
                elif angle < 65:
                    self.feedback_status = "Lower down your knee."
             
            if angle > 160 and self.stage == "down":
                self.sequence_stage = "Straighten Legs"
                self.feedback_status = "Legs straightened. Proceed to curl."

     

            # Ensure the left elbow landmark is detected
            if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.5: 
                # Get the left elbow landmark coordinates
                right_knee_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image.shape[1])
                right_knee_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image.shape[0])

                # Display the angle value near the left elbow position
                cv2.putText(image, str(angle), 
                            (right_knee_x, right_knee_y),  # Use left elbow coordinates
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"Stage: {self.sequence_stage}",
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
        cv2.putText(image, self.stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        return av.VideoFrame.from_ndarray(image, format='bgr24')
    
    def get_feedback_status(self):
        return self.feedback_status

def app():
    st.set_page_config(layout="wide")
    st.title("Right High Knee Exercise")
    remind_text = "**Make sure to allow access of your camera and speaker. Refresh the website if there is a lag.**"
    st.markdown(remind_text)
    remark_text = "**Perfect angle for a high knee is 85 degree to 125 degree. Try now with your right leg! Make sure to show your lower/full body with your right leg (side) into your webcam.**"
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
            if feedback_status == "Legs straightened. Proceed to curl.":
                if feedback_status != last_feedback:
                    last_feedback = feedback_status
            elif feedback_status == "Good High Knee!":
                filename = 'good_high_knee.mp3'
                if feedback_status != last_feedback:
                    play_audio(feedback_status, filename)
                    last_feedback = feedback_status
            elif feedback_status == "Lower down your knee.":
                filename = 'lower_knee.mp3'
                if feedback_status != last_feedback:
                    play_audio(feedback_status, filename)
                    last_feedback = feedback_status


if __name__ == '__main__':
    app()
