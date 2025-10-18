# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amir Mohammadi  <amir.mohammadi@idiap.ch>
#
# SPDX-License-Identifier: MIT

import io

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from model import Cunet, preprocess_image
from traceback import format_exception
from fastapi.responses import Response

app = FastAPI(title="Fake Image Detection API")

MODEL = Cunet(model_path='/app/src/Cue_Net/ck/UNetModel-epoch=31-f1=0.9805.pt',device='cuda')


@app.post("/detect")
def detect(image: UploadFile):
    """Returns a single float score for the image.
    The score is close to 1 for real images and close to 0 for forged images.
    A threshold of 0.5 is used to classify images as real or forged.
    """
    try:
        # Load image (Do not change this line)
        img = np.array(Image.open(image.file).convert("RGB"))

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it,
        # and return a single float score for the image
        img = preprocess_image(img)
        score = MODEL.detect(img)
        # *** Your code goes above here ***

        # Do not change the lines below
        return validate_score(score)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process the image: {format_exception(e)}",
        )


def validate_score(score):
    score = float(score)
    if score < 0 or score > 1:
        raise RuntimeError(f"Score {score} is not in the range [0, 1]")
    return {"score": score}


@app.post(
    "/localize",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
def localize(image: UploadFile):
    """Returns a binary mask of the image."""
    try:
        # Load image (Do not change this line)
        img = np.array(Image.open(image.file).convert("RGB"))
        image_size = img.shape[:2]

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it, and
        # return a binary mask of the image
        img = preprocess_image(img)
        mask = MODEL.localize(img)
        # *** Your code goes above here ***

        # Do not change the following lines
        return validate_mask(image_size, mask)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process the image: {format_exception(e)}",
        )


def validate_mask(image_size, mask, **kwargs):
    if not mask.shape == image_size:
        raise RuntimeError(
            f"Mask size {mask.shape} does not match image size {image_size}"
        )
        # make sure mask is binary
    if not all(np.isin(np.unique(mask), [False, True])):
        raise RuntimeError(f"Mask is not binary! Found values: {np.unique(mask)}")

    mask = Image.fromarray(mask)
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    return Response(content=mask_bytes.getvalue(), media_type="image/png", **kwargs)


@app.post("/detect_and_localize", response_model=dict)
def detect_and_localize(image: UploadFile):
    """Returns a single float score for the image and a binary mask of the image.
    Implementation of this function is optional but highly recommended.
    Implement this function if you are using the same model for both detection and
    localization.
    """
    try:
        # Load image (Do not change this line)
        img = np.array(Image.open(image.file).convert("RGB"))
        image_size = img.shape[:2]

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it, and
        # return a binary mask of the image
        img = preprocess_image(img)
        score, mask = MODEL.detect_and_localize(img)
        # *** Your code goes above here ***

        # Do not change the following lines
        score = validate_score(score)
        return validate_mask(
            image_size, mask, headers={"X-Score-Value": str(score["score"])}
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process the image: {format_exception(e)}",
        )
