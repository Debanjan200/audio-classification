from flask import Flask,render_template,request
from keras.models import load_model
import os
import numpy as np
import librosa
import warnings

app=Flask(__name__)

def feature_extractor(file_name):
    warnings.simplefilter("ignore")
    audio_data,sample_rate=librosa.load(file_name)
    mfccs=librosa.feature.mfcc(y=audio_data,sr=sample_rate,n_mfcc=50)
    mfccs=np.mean(mfccs.T,axis=0)
    mfccs=np.reshape(mfccs,(1,50))
    return mfccs

def prediction(data):
    model=load_model("hindi_speech_prediction_model.h5")
    predict=model.predict(data)
    return predict[0][0]



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        target_img=os.path.join(os.getcwd(),'uploads')
        f=request.files["my_audio"]

        f.save(os.path.join(target_img,f.filename))
        audio_path=os.path.join(target_img,f.filename)
        print(audio_path)
        audio=feature_extractor(audio_path)
        predict=prediction(audio)

        st=""
        if predict>0.5:
            st="Male"

        else:
            st="Female"

        return render_template("index.html",st=st,prediction=predict)


if __name__=="__main__":
    app.run(debug=True)