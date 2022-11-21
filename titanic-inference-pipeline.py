import modal
    
LOCAL=False

#os.environ["HOPSWORKS_API_KEY"] = "..."

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image","xgboost"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    # Get the latest inserted batch data element
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    passenger = y_pred[y_pred.size-1]
    dataset_api = project.get_dataset_api()

    print(f'Survival predicted: {passenger}')

    smiley_url = "https://raw.githubusercontent.com/Neproxx/titanic-pipeline/main/assets/" + str(passenger) + ".jpg"
    img = Image.open(requests.get(smiley_url, stream=True).raw)
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)
    ## PREDICTION COMPLETE

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()

    # get the label from the last added element
    label = df.iloc[-1]["survived"]
    if label == 0:
        label_name = "Dead"
    else:
        label_name = "Survived"

    print("Actual Survivial status: " + label_name)
    label_url = "https://raw.githubusercontent.com/Neproxx/titanic-pipeline/main/assets/" + str(label) + ".jpg"
    img_raw = requests.get(label_url, stream=True).raw
    img = Image.open(img_raw)
    img.save("./actual_passenger.png")
    dataset_api.upload("./actual_passenger.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Passenger Survival Prediction Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [passenger],
        'label': [label],
        'datetime': [now],
       }

    # add prediction with relevant metadata to monitor feature group
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    # Upload picture of recents for some reason
    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when enough predictions are available
    print("Number of predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Dead', 'True Survivor'],
                             ['Pred Dead', 'Pred Survivor'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different passenger predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different passenger predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

