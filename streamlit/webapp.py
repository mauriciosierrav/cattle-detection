import io
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
    - Select the model you want to use for prediction (best.pt is the model re-trained by the developers).
    - Click on "Get Prediction" and get the inference and a summary of the results.

    This web app uses a FastAPI service as a backend. Visit this URL http://127.0.0.1:8000/docs for documentation"""
)

# file uploader
input_image = st.file_uploader("Upload an image")

# select model
model_s = st.radio("Select model you want to use", ['best.pt', 'yolov5s.pt', 'yolov5l.pt'])
if model_s == 'best.pt':
    params = {'weights': 'best.pt'}
    st.write('The selected model is: best.pt')
elif model_s == 'yolov5s.pt':
    params = {'weights': 'yolov5s.pt'}
    st.write('The selected model is: yolov5s.pt')
elif model_s == 'yolov5l.pt':
    params = {'weights': 'yolov5l.pt'}
    st.write('The selected model is: yolov5.pt')


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
        rem_list = ['server', 'content-length', 'content-type']
        for key in rem_list:
            del headers_[key]
        json_headers = json.dumps(headers_)
        st.json(json_headers)
    else:
        # handle case with no image
        st.write("Insert an image!")
