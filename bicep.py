# import cv2
# import mediapipe as mp
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
# from gtts import gTTS
# import base64

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

# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.counter = 0
#         self.stage = "down"
#         self.sequence_stage = "Initial"
#         self.feedback_status = ""
#         self.last_feedback = ""

#     def play_audio(self, feedback_text, filename):
#         tts = gTTS(feedback_text)
#         tts.save(filename)
#         audio_file = filename

#         b64_data = autoplay_audio(audio_file)
#         html_content = create_audio_html(b64_data)
#         st.markdown(html_content, unsafe_allow_html=True)

#     def transform(self, frame):
#         image = frame.to_ndarray(format="bgr24")

#         with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#             results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#             try:
#                 landmarks = results.pose_landmarks.landmark

#                 shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 angle = calculate_angle(shoulder, elbow, wrist)

#                 if angle > 160 and self.sequence_stage != "Bicep Curl":
#                     self.sequence_stage = "Straighten Arms"
#                     self.stage = "down"
#                     self.feedback_status = "Arms straightened. Go to curl."
#                 elif self.sequence_stage == "Straighten Arms" and angle < 110 and self.stage == "down":
#                     self.sequence_stage = "Bicep Curl"
#                     self.stage = "up"
#                     self.counter += 1
#                     print(self.counter)

#                 if self.sequence_stage == "Bicep Curl":
#                     if angle < 45:
#                         self.feedback_status = "Lower down your arm."
#                         filename = 'lower_arm.mp3'
#                     elif 45 <= angle <= 65:
#                         self.feedback_status = "Good Curl!"
#                         filename = 'goodcurl.mp3'
#                     elif angle > 160 and self.stage == "up":
#                         self.sequence_stage = "Straighten Arms"
#                         self.stage = "down"
#                         self.feedback_status = "Arms straightened. Go to curl."
#                         self.last_feedback = self.feedback_status
#                         return image

#                     if self.feedback_status != self.last_feedback:
#                         self.play_audio(self.feedback_status, filename)
#                         self.last_feedback = self.feedback_status

#                 if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5:
#                     left_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1])
#                     left_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])

#                     cv2.putText(image, str(angle),
#                                 (left_elbow_x, left_elbow_y),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 cv2.putText(image, f"Stage: {self.sequence_stage}",
#                             (10, image.shape[0] - 55),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#                 cv2.putText(image, f"Feedback: {self.feedback_status}",
#                             (10, image.shape[0] - 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#             except Exception as e:
#                 print(e)
#                 pass

#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
#             cv2.putText(image, 'REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 1, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(self.counter), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, 'STAGE', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, self.stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                                       mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

#             image = cv2.resize(image, (1200, 800))

#         return image

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

#     # Create a sidebar
#     st.sidebar.title('Left Bicep Curl Exercise')
#     st.sidebar.subheader('Parameters')
#     # Creating a button for webcam
#     use_webcam = st.sidebar.button('Use Webcam')
#     st.markdown(' ## Output')
#     stframe = st.empty()

#     if use_webcam:
#         gif_placeholder.empty()  # Clear the GIF
#         webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
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

import streamlit as st
import cv2
import numpy as np

# Function to process video frames with OpenCV
def process_frame(frame):
    # Convert frame to OpenCV format (BGR)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform some processing with OpenCV
    # For example, apply a simple edge detection filter
    edges = cv2.Canny(frame_bgr, 100, 200)

    # Convert processed frame back to RGB format for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_rgb

# Streamlit app
st.title("OpenCV Video Frame Processing")

# HTML and JavaScript code to access webcam and send stream
html_code = """
<video id="localVideo" autoplay playsinline muted></video>
<script>
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      const videoElement = document.getElementById('localVideo');
      videoElement.srcObject = stream;
    })
    .catch(error => {
      console.error('Error accessing webcam:', error);
    });
</script>
"""

# Render HTML code in Streamlit app
st.components.v1.html(html_code, height=400)

# Placeholder for displaying processed frames
processed_frame_placeholder = st.empty()

# Continuously process frames from the webcam stream
while True:
    # Capture frame from webcam stream
    # Here, you would retrieve frames from your 'offer' endpoint
    # For demonstration purposes, we'll create a dummy frame
    dummy_frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Process frame with OpenCV
    processed_frame = process_frame(dummy_frame)

    # Display processed frame in Streamlit app
    processed_frame_placeholder.image(processed_frame, channels="BGR")

    # Add a short delay to control frame rate
    # This is optional and can be adjusted as needed
    # Depending on your application, you may want to synchronize frame processing with the actual frame rate
    # For real-time applications, you may want to remove this delay
    st.experimental_streamlit_sleep(0.1)

