import asyncio
import websockets
import cv2
import numpy as np
import base64
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

async def video_handler(websocket, path):
    while True:
        try:
            # Receive base64 encoded frame from the client
            data = await websocket.recv()
            frame_data = base64.b64decode(data)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Process the frame with MediaPipe
            frame = process_frame(frame)

            # Encode the frame back to base64 and send it to the client
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(encoded_frame)
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

def process_frame(frame):
    # Convert the image from BGR to RGB as MediaPipe uses RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe to process the frame
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame_rgb, detection)

    # Convert the image back from RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr

start_server = websockets.serve(video_handler, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
