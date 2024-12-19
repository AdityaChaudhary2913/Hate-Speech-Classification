from flask import Flask, render_template, url_for, request, session, jsonify, redirect
from hate.exception import CustomException
from hate.constants import *
import sys
import os
from joblib import load
from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from pymongo import MongoClient


app = Flask(__name__)
app.secret_key = os.getenv("SessionSecretKey")

# MongoDB Atlas connection
client = MongoClient(os.getenv("MONGODB_URL") )
db = client['hate_speech']

ADMIN_ID = os.getenv("AdminID")
ADMIN_PASSWORD = os.getenv("AdminPassword")

try:    
    model = load('trained_model/model.pkl')
except Exception as e:
    model = None

@app.route("/")
def home():
    return render_template('home.html', name=model)

@app.route("/train")
def train_route():
    if not session.get('admin_logged_in'):
        return redirect(url_for('home')) 
    try:
        if not model:
            train_pipeline = TrainPipeline()
            train_pipeline.run_pipeline()
        return render_template('training.html')
    except Exception as e:
        raise CustomException(e,sys)


@app.route("/admin_login", methods=['POST'])
def admin_login():
    data = request.get_json()
    admin_id = data.get('adminID')
    admin_password = data.get('adminPassword')
    if admin_id == ADMIN_ID and admin_password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return jsonify(success=True)
    else:
        session['admin_logged_in'] = False
        return jsonify(success=False)

@app.route("/logout")
def logout():
    session['admin_logged_in'] = False
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

@app.route("/text_classifier", methods=['POST', 'GET'])
def text_classifier():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify(error="Please enter a text"), 400
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(text)
        result = 'Hate' if prediction == "hate and abusive" else 'No Hate'
        return jsonify(result=result, text=text)
    return render_template('text_classifier.html')
    
if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug= True)








# from hate.pipeline.train_pipeline import TrainPipeline
# train_pipeline = TrainPipeline()
# train_pipeline.run_pipeline()

# from hate.pipeline.prediction_pipeline import PredictionPipeline
# prediction_pipeline = PredictionPipeline()
# prediction_pipeline.run_pipeline("Hello boy!")

# from pymongo import MongoClient
# import gridfs
# from bson import ObjectId
# import os
# from tensorflow.keras.models import load_model

# client = MongoClient('mongodb+srv://aditya1306:aditya1306@phishingclassifier.y2fj1x3.mongodb.net/')
# db = client['hate_speech']
# fs = gridfs.GridFS(db)

# # Function to save a directory to GridFS
# def save_directory_to_gridfs(directory_path, fs, model_name):
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             with open(file_path, 'rb') as f:
#                 data = f.read()
#                 fs.put(data, filename=f"{model_name}/{os.path.relpath(file_path, directory_path)}")

# # Function to download a model directory from GridFS
# def download_directory_from_gridfs(model_name, fs, target_path):
#     os.makedirs(target_path, exist_ok=True)
#     for grid_out in fs.find({'filename': {'$regex': f'^{model_name}/'}}):
#         file_path = os.path.join(target_path, os.path.relpath(grid_out.filename, model_name))
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, 'wb') as f:
#             f.write(grid_out.read())

# # Upload the entire model directory to GridFS
# model_dir = './model'  # Directory where the SavedModel format is saved
# model_name = 'saved_model'
# save_directory_to_gridfs(model_dir, fs, model_name)

# # Download the model directory from GridFS
# downloaded_dir = './downloaded_model'
# download_directory_from_gridfs(model_name, fs, downloaded_dir)

# # Load the model from the downloaded directory
# try:
#     model = load_model(downloaded_dir)
#     print("Model loaded successfully from downloaded data.")
# except Exception as e:
#     print("Failed to load model from downloaded data:", e)



















# # Upload model
# with open('./notebook/model.h5', 'rb') as f:
#     contents = f.read()
# stored_model_id = fs.put(contents, filename='model.h5')

# # Check upload by downloading immediately
# contents_downloaded = fs.get(stored_model_id).read()
# with open('downloaded_model.h5', 'wb') as f:
#     f.write(contents_downloaded)

# # Compare sizes
# print("Original size:", len(contents))
# print("Downloaded size:", len(contents_downloaded))

# import h5py

# def print_model_details(file_path):
#     with h5py.File(file_path, 'r') as file:
#         print("Keys:", list(file.keys()))
#         print("Model config:", file.attrs['model_config'])
#         print("Training config:", file.attrs['training_config'])

# print_model_details('./notebook/model.h5')
# print_model_details('downloaded_model.h5')

# try:
#     model = load_model('downloaded_model.h5')
#     print("Model loaded successfully from downloaded data.")
# except Exception as e:
#     print("Failed to load model from downloaded data:", e)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# with open('./notebook/model.h5', 'rb') as f:
#     model_data = f.read()
# model_id = fs.put(model_data)

# model_data = fs.get(ObjectId("6763af2ebf7b7ff74451d48c")).read()
# with open('downloaded_model.h5', 'wb') as file:
#     file.write(model_data)

# model = load_model('downloaded_model.h5', compile=False)


# from fastapi import FastAPI
# import uvicorn
# import sys
# from fastapi.templating import Jinja2Templates
# from starlette.responses import RedirectResponse
# from fastapi.responses import Response
# from hate.pipeline.prediction_pipeline import PredictionPipeline
# from hate.exception import CustomException
# from hate.constants import *


# text:str = "What is machine learing?"

# app = FastAPI()

# @app.get("/", tags=["authentication"])
# async def index():
#     return RedirectResponse(url="/docs")




# @app.get("/train")
# async def training():
#     try:
#         train_pipeline = TrainPipeline()

#         train_pipeline.run_pipeline()

#         return Response("Training successful !!")

#     except Exception as e:
#         return Response(f"Error Occurred! {e}")
    


# @app.post("/predict")
# async def predict_route(text):
#     try:

#         obj = PredictionPipeline()
#         text = obj.run_pipeline(text)
#         return text
#     except Exception as e:
#         raise CustomException(e, sys) from e
    



# if __name__=="__main__":
#     uvicorn.run(app, host=APP_HOST, port=APP_PORT)


    

