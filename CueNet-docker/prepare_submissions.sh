#!/bin/bash

# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amir Mohammadi  <amir.mohammadi@idiap.ch>
#
# SPDX-License-Identifier: MIT

# Check if the required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: ./prepare_submission.sh <team_name> <track_name> <algorithm_name> <version>"
    exit 1
fi

# Assign arguments to variables for better readability
TEAM_NAME=$1
TRACK_NAME=$2
ALGORITHM_NAME=$3
VERSION=$4

# Validate TRACK_NAME
if [[ "$TRACK_NAME" != "track_1" && "$TRACK_NAME" != "track_2" && "$TRACK_NAME" != "track_both" ]]; then
    echo "Error: Invalid track name. Must be one of: track_1, track_2, track_both."
    exit 1
fi

# Construct the image name and filename
IMAGE_NAME="${TEAM_NAME}/${TRACK_NAME}_${ALGORITHM_NAME}:${VERSION}"
FILE_NAME="${TEAM_NAME}_${TRACK_NAME}_${ALGORITHM_NAME}_${VERSION}.tgz"

# Display the constructed image name and file name
echo "Docker Image Name: ${IMAGE_NAME}"
echo "Output File Name: ${FILE_NAME}"

# Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}..."
if docker build -t "${IMAGE_NAME}" .; then
    echo "Docker image built successfully."
else
    echo "Error: Failed to build Docker image."
    exit 2
fi

# Save the Docker image to a tarball and compress it
echo "Saving Docker image to file: ${FILE_NAME}..."
if docker save "${IMAGE_NAME}" | gzip >"${FILE_NAME}"; then
    echo "Docker image saved successfully."
else
    echo "Error: Failed to save Docker image to file."
    exit 3
fi

# Remove the Docker image to clean up
echo "Removing Docker image: ${IMAGE_NAME}..."
if docker image rm "${IMAGE_NAME}"; then
    echo "Docker image removed successfully."
else
    echo "Warning: Failed to remove Docker image. You may need to clean up manually."
fi
