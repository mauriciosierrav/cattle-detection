import io
import os
from PIL import Image
from starlette.responses import Response
import torch
from fastapi import FastAPI, File, Query
import numpy as np


# Instantiate FastAPI class
app = FastAPI(
    title="API for Cattle Detection",
    description="Cattle Detection using Ultralytics YOLOv5 pretrained models",
    version="0.1.0",
)


def get_image_from_bytes(binary_image, max_size=400) -> Image:
    """
    :param      binary_image: Image as sequence of bytes
    :type       binary_image: bytes
    :param      max_size: integer defining the maximum size (width or height) of the image
    :type       max_size: int
    :return:    resized RGB image
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


@app.post("/object-to-img")
def get_prediction(file: bytes = File(...), weights: str = Query(default="yolov5s.pt")) -> Response:
    """
    Possible ``weights`` values:
        - yolov5s.pt (default)
        - yolov5l.pt
        - any other model in ./model/

    Parameters
    ----------
    file : bytes
        Image as sequence of bytes

    weights : str
        name of model to use

    Returns
    -------
    content:    Image as a sequence of bytes with the results of the inference

    headers:    confidence, class and name of predictions.
    """
    # Instantiate model from model folder
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    repo_or_dir, path = os.path.join(fileDirectory, f'yolov5'), os.path.join(fileDirectory, f'model/{weights}')

    model = torch.hub.load(repo_or_dir, 'custom', path, source='local')

    # modify the name of the cow class and make it lower case
    names = dict(model.names)
    model.names = {k: 'cattle' if v == 'cow' else str.lower(v) for k, v in names.items()}

    # predictions
    segmented_image = get_image_from_bytes(file)
    predict = model(segmented_image)

    # content
    results_img = Image.fromarray(predict.render()[0].astype(np.uint8))
    bytes_io = io.BytesIO()
    results_img.save(bytes_io, format="PNG")

    # headers
    results_list = predict.pandas()
    result_dict = results_list.xyxyn[0][['confidence', 'class', 'name']].to_dict(orient="index")
    for k, v in result_dict.copy().items():
        result_dict[k]['confidence'] = str(round(result_dict[k]['confidence'] * 100, 1)) + '%'
        chars = ['{', '}', "'"]
        text = str(v)
        for c in chars:
            if c in text:
                text = text.replace(c, '')
        result_dict[k] = text
        result_dict[str(k)] = result_dict.pop(k)

    return Response(content=bytes_io.getvalue(), media_type="image/png", headers=result_dict)
