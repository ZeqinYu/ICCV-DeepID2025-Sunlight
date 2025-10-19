#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amir Mohammadi  <amir.mohammadi@idiap.ch>
#
# SPDX-License-Identifier: MIT

import io
import pytest
import requests
import numpy as np
from PIL import Image
from pathlib import Path

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Path to test images
TEST_IMAGES_DIR = Path("test_images")
PRISTINE_IMAGES = [TEST_IMAGES_DIR / "pristine1.jpg"]
TAMPERED_IMAGES = [TEST_IMAGES_DIR / "tampered1.png"]

def test_server_is_running():
    """Test if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("Server is not running")

def test_detect_endpoint():
    """Test the detect endpoint with both pristine and tampered images."""
    print()
    for image_path in PRISTINE_IMAGES:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/detect", files=files)

            assert response.status_code == 200, (
                f"Failed to detect {image_path.name}. Response was {response.text}"
            )

            data = response.json()
            assert "score" in data, f"Missing score in response for {image_path.name}"

            score = data["score"]
            assert isinstance(score, (int, float)), (
                f"Score is not a number for {image_path.name}"
            )
            print(f"Pristine image {image_path.name} score: {score:.2f}")

    for image_path in TAMPERED_IMAGES:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            response = requests.post(f"{BASE_URL}/detect", files=files)

            assert response.status_code == 200, f"Failed to detect {image_path.name}"

            data = response.json()
            assert "score" in data, f"Missing score in response for {image_path.name}"

            score = data["score"]
            assert isinstance(score, (int, float)), (
                f"Score is not a number for {image_path.name}"
            )

            print(f"Tampered image {image_path.name} score: {score:.2f}")

def test_localize_endpoint():
    """Test the localize endpoint with both pristine and tampered images."""
    print()
    for image_path in PRISTINE_IMAGES + TAMPERED_IMAGES:
        with open(image_path, "rb") as f:
            files = {
                "image": (
                    image_path.name,
                    f,
                    "image/jpeg" if image_path.name.endswith(".jpg") else "image/png",
                )
            }
            response = requests.post(f"{BASE_URL}/localize", files=files)

            assert response.status_code == 200, (
                f"Failed to localize {image_path.name}, response: {response.text}"
            )
            assert response.headers["content-type"] == "image/png", (
                f"Response is not a PNG image for {image_path.name}"
            )

            mask = Image.open(io.BytesIO(response.content))
            assert mask.size[0] > 0 and mask.size[1] > 0, (
                f"Invalid mask size for {image_path.name}"
            )

            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            assert len(unique_values) <= 2, f"Mask is not binary for {image_path.name}"

            if image_path in PRISTINE_IMAGES:
                white_percentage = np.sum(mask_array) / mask_array.size
                print(
                    f"Pristine image {image_path.name} white percentage: {white_percentage:.2%}"
                )
            else:
                black_percentage = np.sum(~mask_array.astype(bool)) / mask_array.size
                print(
                    f"Tampered image {image_path.name} black percentage: {black_percentage:.2%}"
                )

def test_detect_and_localize_endpoint():
    """Test the detect_and_localize endpoint with both pristine and tampered images."""
    print()
    for image_path in PRISTINE_IMAGES + TAMPERED_IMAGES:
        with open(image_path, "rb") as f:
            files = {
                "image": (
                    image_path.name,
                    f,
                    "image/jpeg" if image_path.name.endswith(".jpg") else "image/png",
                )
            }
            response = requests.post(f"{BASE_URL}/detect_and_localize", files=files)

            assert response.status_code == 200, (
                f"Failed to detect_and_localize {image_path.name}, response: {response.text}"
            )
            assert response.headers["content-type"] == "image/png", (
                f"Response is not a PNG image for {image_path.name}"
            )
            assert "X-Score-Value" in response.headers, (
                f"Missing X-Score-Value header for {image_path.name}"
            )
            score = float(response.headers["X-Score-Value"])
            mask = Image.open(io.BytesIO(response.content))
            assert mask.size[0] > 0 and mask.size[1] > 0, (
                f"Invalid mask size for {image_path.name}"
            )
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            assert len(unique_values) <= 2, f"Mask is not binary for {image_path.name}"
            if image_path in PRISTINE_IMAGES:
                white_percentage = np.sum(mask_array) / mask_array.size
                print(
                    f"Pristine image {image_path.name} detect_and_localize score: {score:.2f}, white percentage: {white_percentage:.2%}"
                )
            else:
                black_percentage = np.sum(~mask_array.astype(bool)) / mask_array.size
                print(
                    f"Tampered image {image_path.name} detect_and_localize score: {score:.2f}, black percentage: {black_percentage:.2%}"
                )

def test_api_compliance():
    """Test overall API compliance."""
    # Check if at least one of the required endpoint groups is implemented

    # Group 1: detect
    try:
        detect_response = requests.post(
            f"{BASE_URL}/detect",
            files={
                "image": (
                    PRISTINE_IMAGES[0].name,
                    open(PRISTINE_IMAGES[0], "rb"),
                    "image/jpeg",
                )
            },
        )
        group1_implemented = detect_response.status_code == 200
    except requests.exceptions.ConnectionError:
        group1_implemented = False
    finally:
        # Close the file if it was opened
        if "detect_response" in locals():
            detect_response.close()

    # Group 2: localize
    try:
        localize_response = requests.post(
            f"{BASE_URL}/localize",
            files={
                "image": (
                    PRISTINE_IMAGES[0].name,
                    open(PRISTINE_IMAGES[0], "rb"),
                    "image/jpeg",
                )
            },
        )
        group2_implemented = localize_response.status_code == 200
    except requests.exceptions.ConnectionError:
        group2_implemented = False
    finally:
        # Close the file if it was opened
        if "localize_response" in locals():
            localize_response.close()

    # At least one group must be implemented
    assert group1_implemented or group2_implemented, (
        "Neither of the required endpoint groups is implemented"
    )

    print("API compliance test passed!")

if __name__ == "__main__":
    # Run the tests
    print("Testing API compliance...")

    try:
        test_server_is_running()
        print("✓ Server is running")

        # Test detection endpoints
        try:
            test_detect_endpoint()
            print("✓ Detect endpoint is working")
            detection_implemented = True
        except (AssertionError, requests.exceptions.ConnectionError) as e:
            print(f"✗ Detection endpoints test failed: {e}")
            detection_implemented = False

        # Test localization endpoint
        try:
            test_localize_endpoint()
            print("✓ Localize endpoint is working")
            localization_implemented = True
        except (AssertionError, requests.exceptions.ConnectionError) as e:
            print(f"✗ Localization endpoint test failed: {e}")
            localization_implemented = False

        # Test detect_and_localize endpoint
        try:
            test_detect_and_localize_endpoint()
            print("✓ Detect and localize endpoint is working")
            detect_and_localize_implemented = True
        except (AssertionError, requests.exceptions.ConnectionError) as e:
            print(f"✗ Detect and localize endpoint test failed: {e}")
            detect_and_localize_implemented = False

        # Check overall compliance
        if (
            detection_implemented
            or localization_implemented
            or detect_and_localize_implemented
        ):
            print("✓ API is compliant")
        else:
            print("✗ API is not compliant")

    except Exception as e:
        print(f"✗ Test failed: {e}")
