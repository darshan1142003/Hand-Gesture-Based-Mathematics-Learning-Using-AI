import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Streamlit configuration
st.set_page_config(layout="wide")
st.image('math gestures.png')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.empty()  # Initialize an empty placeholder for the answer

# Configure the Generative AI model
genai.configure(api_key="AIzaSyBLQoSRpZzxLVDJ9QRP4LM_FMw8IfBC-Bg")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    """Find hands in the current frame without drawing landmarks."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)  # Count the number of fingers up for the first hand
        return fingers, lmList
    return None


def draw(info, prev_pos, canvas):
    """Draw lines on canvas based on hand gestures."""
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = (lmList[8][0], lmList[8][1])
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=7)
        prev_pos = current_pos
    else:
        prev_pos = None
    return prev_pos, canvas


def sendToAI(model, canvas, fingers):
    """Send the canvas image to AI for solving the math problem."""
    if fingers == [0, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        return response.text
    return ""


prev_pos = None
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Initialize canvas with the same size as the video frame
output_text = ""  # Initialize output_text here

# Continuously get frames from the webcam
while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from webcam.")
        break

    # Resize the frame if necessary to match the output window size
    img_resized = cv2.resize(img, (1280, 720))  # Adjust size as needed

    img_resized = cv2.flip(img_resized, flipCode=1)

    info = getHandInfo(img_resized)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

        # Clear canvas when thumb is up
        if fingers == [1, 0, 0, 0, 0]:  # Thumb up
            canvas = np.zeros_like(img_resized)

    # Combine the image and canvas for display
    image_combined = cv2.addWeighted(img_resized, 0.7, canvas, 0.3, 0)

    # Display the combined image in the Streamlit app
    FRAME_WINDOW.image(image_combined, channels="BGR", use_column_width=True)

    if output_text:
        output_text_area.text(output_text)

# Release resources
cap.release()
cv2.destroyAllWindows()