@echo off
cd Hellen_model_RN
call venv\Scripts\activate
py inference_classifier.py
cmd /k
