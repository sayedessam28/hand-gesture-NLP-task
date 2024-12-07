#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import streamlit as st
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np

# Initialize NLP models
summarizer = pipeline("summarization", model="t5-small")
translator_to_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Function to detect gestures
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    if index_tip.y < thumb_tip.y:
        return "Swipe Up"
    elif index_tip.y > thumb_tip.y:
        return "Swipe Down"
    return None

# Streamlit App Layout
st.title("Gesture-Controlled NLP App")
st.write("Perform gestures to control NLP tasks:")
st.write("- **Swipe Up**: Summarize text.")
st.write("- **Swipe Down**: Translate text to Arabic.")

# Input Text Area
input_text = st.text_area("Enter Text", placeholder="Type or paste your text here...", height=150)

# Result Display
result_placeholder = st.empty()

# Streamlit WebRTC Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands()
        self.last_gesture = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        gesture_detected = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_detected = detect_gesture(hand_landmarks.landmark)

        # Perform NLP tasks based on detected gesture
        global input_text
        if gesture_detected:
            if gesture_detected == "Swipe Up" and input_text:
                summary = summarizer(input_text, max_length=50, min_length=25, do_sample=False)
                result = summary[0]['summary_text']
                self.last_gesture = "Summarize"
                st.session_state['result'] = f"**Summary:** {result}"
            elif gesture_detected == "Swipe Down" and input_text:
                translation = translator_to_ar(input_text)
                result = translation[0]['translation_text']
                self.last_gesture = "Translate"
                st.session_state['result'] = f"**Translation (Arabic):** {result}"

        # Display landmarks in the video feed
        return frame.from_ndarray(img, format="bgr24")

# Initialize Result State
if 'result' not in st.session_state:
    st.session_state['result'] = "Waiting for gesture..."

# Display Result
result_placeholder.markdown(st.session_state['result'])

# Webcam Stream
webrtc_streamer(key="gesture-control", video_processor_factory=VideoProcessor)


# In[ ]:




