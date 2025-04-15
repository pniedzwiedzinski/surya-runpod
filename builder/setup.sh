#!/bin/bash

# NOTE: This script is not ran by default for the template docker image.
#       If you use a custom base image you can add your required system dependencies here.

set -e # Stop script on error
apt-get update && apt-get upgrade -y # Update System

python3.11 -c "from surya.models import load_predictors; load_predictors()"

# Clean up, remove unnecessary packages and help reduce image size
apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
