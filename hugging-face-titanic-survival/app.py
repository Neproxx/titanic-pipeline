import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib
import xgboost as xgb

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=3) # NOTE: Choose version as needed
model_dir = model.download()
# model = joblib.load(model_dir + "/titanic_model.pkl")
model = xgb.XGBClassifier()
model.load_model(model_dir + "/titanic_model.json") # xgboost has problems with pickle


def titanic(passenger_class, sex, age, fare):
    input_list = []
    input_list.append(pclass_mapping[passenger_class])
    input_list.append(sex_mapping[sex])
    input_list.append(age_mapping[age])
    input_list.append(int(fare))
    
    # access with [0] because predict returns a list of predictions
    y_pred = model.predict(np.asarray(input_list).reshape(1, -1))[0]
    smiley_url = "https://raw.githubusercontent.com/Neproxx/titanic-pipeline/main/assets/" + str(y_pred) + ".jpg"
    img = Image.open(requests.get(smiley_url, stream=True).raw)            
    return img
        
sex_mapping = {"Male": 0, "Female": 1}
pclass_mapping = {
    "Upper Class": 3,
    "Middle Class": 2,
    "Lower Class": 1
}
age_mapping = {
    "Unknown": 0,
    "Child": 1,
    "Teenager": 2,
    "Young Adult": 3,
    "Adult": 4,
    "Senior": 5
}

demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Specify a passenger and predict whether he would have survived.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(default="Middle Class", choices=[k for k in pclass_mapping.keys()], label="Passenger Class"),
        gr.inputs.Dropdown(default="Female", choices=[k for k in sex_mapping.keys()], label="Sex"),
        gr.inputs.Dropdown(default="Child", choices=[k for k in age_mapping.keys()], label="Age"),
        gr.inputs.Slider(minimum=1, maximum=10, default=8, step=1, label="ticket price category"),
        # add integer slider


        ],
    outputs=gr.Image(type="pil"))

demo.launch()

