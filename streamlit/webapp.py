import io
import os
import requests
from PIL import Image
import streamlit as st
import json

# url
url = "http://127.0.0.1:8000/object-to-img"

# title
st.title("Cattle Detection")

# description and instructions
st.write(
    """ 
    #### Get cattle detection for an image, using Ultralytics YOLOv5 pretrained models.

    Instructions for use:
    - Upload an image from your computer
    - Select the model you want to use for prediction
    - Click on "Get Prediction" and get the inference and a summary of the results.

    This web app uses a FastAPI service as a backend. Visit this URL http://127.0.0.1:8000/docs for documentation"""
)

# file uploader
input_image = st.file_uploader("Upload an image")

# models
path_models = '../fastapi/model/'
models = [file for file in os.listdir(path_models) if str.lower(os.path.splitext(file)[1]) == '.pt']

# select model
i = models.index('yolov5s.pt')
model_s = st.radio("Select model you want to use", models, index=i)
params = {'weights': model_s}
st.write(f'The selected model is: {model_s}')


def process(image, server_url):
    """
    :param      image: Image as sequence of bytes
    :type       image: bytes
    :param      server_url: API url
    :type       server_url: str
    :return:    content: Image as a sequence of bytes with the results of the inference
                headers: confidence, class and name of predictions.
    :rtype:     Response
    """
    files = {'file': ('filename', image, 'image/jpeg')}
    r = requests.post(server_url, files=files, params=params)
    return r


# button Get Prediction
if st.button("Get Prediction"):
    col1, col2 = st.columns(2)

    if input_image:
        segments = process(input_image, url)
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")

        # show original image and image with predictions
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Detections")
        col2.image(segmented_image, use_column_width=True)

        # show result as json
        headers_ = dict(segments.headers)
        headers_['execution_date'] = headers_.pop('date')
        headers_ = {k: v for k, v in segments.headers.items() if k not in ['server', 'content-length', 'content-type']}
        st.json(json.dumps(headers_))
    else:
        # handle case with no image
        st.write("Insert an image!")
