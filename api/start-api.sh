#!/bin/bash
# Start API with model path set to /model/saved_model.pth
# NOTE: If you update the model files, run ../../build-dockers.sh to refresh /api/model before starting the API.
export MODEL_PATH=/model/saved_model.pth
python api.py
