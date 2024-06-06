# import cv2
# import mediapipe as mp
# import numpy as np
# import streamlit as st
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

# # HTML and JavaScript code to access webcam and send stream
# html_code = """
# <video id="localVideo" autoplay playsinline muted></video>
# <script>
#   navigator.mediaDevices.getUserMedia({ video: true })
#     .then(stream => {
#       const videoElement = document.getElementById('localVideo');
#       videoElement.srcObject = stream;
#     })
#     .catch(error => {
#       console.error('Error accessing webcam:', error);
#     });
# </script>
# """

# # Render HTML code in Streamlit app
# st.components.v1.html(html_code, height=400)

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
#     # st.set_page_config(layout="wide")  # Set the layout to wide
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

#         cap = cv2.VideoCapture(0)

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
#             while cap.isOpened():
#                 ret, frame = cap.read()
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

#         cap.release()

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

# import asyncio
# import base64
# import cv2
# import numpy as np
# import websockets

# # HTML/JavaScript code
# html_code = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Webcam Stream</title>
# </head>
# <body>
#     <video id="localVideo" autoplay playsinline muted></video>
#     <script>
#         const ws = new WebSocket('ws://localhost:8765');

#         const videoElement = document.getElementById('localVideo');
#         navigator.mediaDevices.getUserMedia({ video: true })
#             .then(stream => {
#                 videoElement.srcObject = stream;
#                 const canvas = document.createElement('canvas');
#                 const ctx = canvas.getContext('2d');
#                 setInterval(() => {
#                     ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
#                     canvas.toBlob(blob => {
#                         ws.send(blob);
#                     }, 'image/jpeg');
#                 }, 1000 / 30);
#             })
#             .catch(error => {
#                 console.error('Error accessing webcam:', error);
#             });
#     </script>
# </body>
# </html>
# """

# # OpenCV processing function
# async def process_frames(websocket, path):
#     while True:
#         # Receive frame from the frontend
#         frame_blob = await websocket.recv()
#         frame = cv2.imdecode(np.frombuffer(frame_blob, np.uint8), cv2.IMREAD_COLOR)

#         # Perform pose estimation and bicep curl exercise detection
#         # Replace this with your actual detection logic using OpenCV

#         # Example: Convert frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Example: Draw a rectangle on the frame
#         cv2.rectangle(gray_frame, (100, 100), (200, 200), (0, 255, 0), 2)

#         # Convert processed frame back to JPEG format and send to frontend
#         _, jpeg_frame = cv2.imencode('.jpg', gray_frame)
#         await websocket.send(base64.b64encode(jpeg_frame).decode('utf-8'))

# async def main():
#     # Start WebSocket server for communication
#     async with websockets.serve(process_frames, "localhost", 8765):
#         # Serve HTML/JavaScript code
#         while True:
#             await asyncio.sleep(1)  # Keep the event loop running

# # Run the asyncio event loop
# if __name__ == "__main__":
#     # Start the main coroutine
#     asyncio.run(main())


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Streamlit app
st.title("MediaPipe Detection with Webcam")

# MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# WebRTC video streaming with higher resolution
webrtc_ctx = webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={
        "video": {
            "width": 1280,  # Set the desired width
            "height": 720,  # Set the desired height
        },
        "audio": False,
    },
)

# Process frames from the video stream
while True:
    if webrtc_ctx.video_receiver:
        try:
            frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
        except queue.Empty:
            frame = None

        if frame is not None:
            img = frame.to_ndarray(format="bgr24")

            # Flip the image horizontally for a later selfie-view display
            img = cv2.flip(img, 1)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img)

            # Draw the face detection annotations on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(img, detection)

            # Display the image in Streamlit
            st.image(img, channels="BGR")
    else:
        break

# Close the MediaPipe face detection
face_detection.close()
