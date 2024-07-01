import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image


TRAINED_MODEL_PATH = 'F:/AIO/helmetdetectionyolov10/model/yolov10best_20240629.pt'
CONF_DEFAULT = 0.25
model = YOLO( TRAINED_MODEL_PATH )

st.title("This is a simple UI YOLOv10 - Helmet Detection")

# Build side bar where user can choose some parameters:
with st.sidebar.title("Configuration:"):
    conf = st.slider("Confidence Score:", 0.0, 1.0, value=CONF_DEFAULT, step=0.1)

with st.form("img_upload"):
    img = st.file_uploader("Select the image:", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Submit")
    
    if submitted and img is not None:
        st.write(f"Image Name: {img.name} - Image Type: {img.type} - Image Size: {img.size/1024: .0f} KB ")
        # Read the image
        image = Image.open(img)
        image_np = np.array(image)

        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)

        if img.type == 'image/jpeg':
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model.predict(source=image_np, conf=conf)

        img_result = results[0].plot()
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        img_show = Image.fromarray(img_result)
        st.image(img_show)

        # # Draw bounding boxes on the image
        # for result in results:
        #     for box in result.boxes:
        #         # Get bounding box coordinates and class
        #         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        #         cls = box.cls

        #         # Draw rectangle
        #         cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #         # Put the score
        #         score = float(box.conf.cpu().numpy())

        #         # Put label
        #         label = f"{model.names[int(cls)]}: {score:.2f}"
        #         cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # Convert the image from BGR to RGB (since OpenCV uses BGR format)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # # Display the image with bounding boxes
        # st.image(image_np, caption='Processed Image.', use_column_width=True)

        # # Optionally, display detection details
        st.write(f"The number of objects detected: {len(results[0].boxes)}")
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls)
                score = float(box.conf.cpu().numpy())
                st.write(f"Class: {model.names[cls]}, Confidence: {score:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")




