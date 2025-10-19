<!--
SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: Amir Mohammadi  <amir.mohammadi@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Demo and Baseline for DeepID 2025 challenge in ICCV 2025

This repository is both a baseline and a code that can be used to submit your own solution for the ICCV 2025 Challenge on Detecting Synthetic Manipulations in ID Documents (DeepID 2025).
Participants can use this code to either run the baseline or modify this code to integrate their own model and build a docker image for submission.

## Testing the baseline

Assuming you are running this code on computer with nvidia GPU and docker, you can run [TruFor](https://github.com/grip-unina/TruFor) as a baseline through the API. Start the model API with:

```bash
docker compose up -d --build
```

and then you can test the API with:

```bash
pytest -sv test_api.py
```

You will need a Python environment with `pytest requests numpy pillow` installed to run the tests. This will be the output of the tests for TruFor:

```text
===================================================================================== test session starts =====================================================================================
platform linux -- Python 3.11.12, pytest-8.3.5, pluggy-1.5.0 -- .pixi/envs/test/bin/python3.11
cachedir: .pytest_cache
rootdir: .
collected 5 items                                                                                                                                                

test_api.py::test_server_is_running PASSED
test_api.py::test_detect_endpoint 
Pristine image pristine1.jpg score: 0.83
Pristine image pristine2.jpg score: 0.82
Tampered image tampered1.png score: 0.00
Tampered image tampered2.png score: 0.00
PASSED
test_api.py::test_localize_endpoint 
Pristine image pristine1.jpg white percentage: 99.27%
Pristine image pristine2.jpg white percentage: 93.51%
Tampered image tampered1.png black percentage: 11.12%
Tampered image tampered2.png black percentage: 71.09%
PASSED
test_api.py::test_detect_and_localize_endpoint 
Pristine image pristine1.jpg detect_and_localize score: 0.83, white percentage: 99.27%
Pristine image pristine2.jpg detect_and_localize score: 0.82, white percentage: 93.51%
Tampered image tampered1.png detect_and_localize score: 0.00, black percentage: 11.12%
Tampered image tampered2.png detect_and_localize score: 0.00, black percentage: 71.09%
PASSED
test_api.py::test_api_compliance API compliance test passed!
PASSED

======================================================================= 5 passed in 9.44s ========================================================================
```

## Submitting your own Model to DeepID Challenge

You can modify this repository to integrate your own model in the API.
Both competition tracks (*detection* and *localization*) are implemented in the same docker container but you can implement only one of the tracks.
Start by removing the TruFor folder [`src/trufor`](src/trufor/) and implement your model in [`src/model.py`](src/model.py).

### Detection Track

To implement the detection track, you need to modify the following `TODO` part in [`src/main.py`](src/main.py):

```python
@app.post("/detect")
def detect(image: UploadFile):
        ...

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it,
        # and return a single float score for the image
        img = preprocess_image(img)
        score = MODEL.detect(img)
        # *** Your code goes above here ***
```

and completely remove the `localize` function if you are not implementing the localization track:

```python
# remove all this code if you are not participating in the localization track

@app.post(
    "/localize",
    ...
)
def localize(image: UploadFile):
    try:
        ...
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process the image: {format_exception(e)}",
        )
```

### Localization Track

To implement the localization track, you need to modify the following `TODO` part in [`src/main.py`](src/main.py):

```python
@app.post(
    "/localize",
    ...
)
def localize(image: UploadFile):
    try:
        ...

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it, and
        # return a binary mask of the image
        img = preprocess_image(img)
        mask = MODEL.localize(img)
        # *** Your code goes above here ***

        ...
```

and completely remove the `detect` function if you are not implementing the detection track.

### Joint Detection and Localization Endpoint (Optional but Recommended)

If you have one model for both detection and localization tasks, you should implement the `/detect_and_localize` endpoint. This is optional but highly recommended if your model supports both tasks as each image will be processed only once.

To implement the detection and localization endpoint, modify the following `TODO` part in [`src/main.py`](src/main.py):

```python
@app.post("/detect_and_localize", response_model=dict)
def detect_and_localize(image: UploadFile):
    try:
        ...

        # TODO: Your code goes below here ***
        # This is where you would preprocess the image, run inference on it, and
        # return a score and a binary mask of the image
        img = preprocess_image(img)
        score, mask = MODEL.detect_and_localize(img)
        # *** Your code goes above here ***

        ...
```

This endpoint should return a boolean PNG image mask as the response body, with the detection score included in the `X-Score-Value` response header. The score should be a float between 0 and 1, where values close to 1 indicate a real image and values close to 0 indicate a tampered image. The mask should be a binary image (same format as in the localization track).

### Fix your requirements.txt and Dockerfile

Change the `requirements.txt` file to include your model dependencies. You can remove the `trufor` dependencies:

```text
# server dependencies
fastapi[standard]==0.115.12
python-multipart==0.0.20
numpy==2.2.2
pillow==11.0.0

# trufor dependencies
torch==2.6.0
timm==1.0.15
```

Do not remove the server dependencies, as they are required to run the API. You can change the versions of the server dependencies if needed.

Finally, you need to change the `Dockerfile` to include your model weights:

```dockerfile
# TODO: Copy the weights here, for example:
COPY weights weights
```

### Testing the model

Once you have integrated your own model, you can test it using the same instruction as in [Testing the baseline](#testing-the-baseline).

### Submitting the docker image

Once you have tested your model, you can submit the docker image to the competition. You can do this by running the following command:

```bash
./prepare_submissions.sh <team_name> <track_name> <algorithm_name> <version>
```

- `<track_name>` must be one of: `track_1` (detection), `track_2` (localization), or `track_both` (detection and localization).

This will create a docker image with the name `<team_name>_<track_name>_<algorithm_name>_<version>.tgz` that you can submit to the competition.

## Issues

If you have any issues with the code, please open an issue in the repository. We will try to help you as soon as possible.
