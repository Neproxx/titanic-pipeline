import os
import modal

LOCAL=False
modal_secret_name = "HOPSWORKS_API_KEY" # alternatives: "hopsworks" "HOPSWORKS_API_KEY"

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib", "scikit-learn", "xgboost"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(modal_secret_name))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    import xgboost as xgb

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login()
    fs = project.get_feature_store()

    # A feature view is a set of features, it is a virtual view that can reference many feature groups.
    # It can define transformations that should be applied to the features before retrieving the data for training.
    # A feature group is a table that actually contains the data.
    try:
        feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    except:
        titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
        query = titanic_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal",
                                          version=1,
                                          description="Read from Titanic dataset",
                                          labels=["survived"],
                                          query=query)

    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train.values.ravel())

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)

    # Create confusion matrix to save
    df_cm = pd.DataFrame(results, ['Actual survived', 'Actual deceased'],
                                  ['Predicted survived', 'Predicted deceased'])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    # Write model binary and confusion matrix to titanic_model folder which will be uploaded later
    model_dir="titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    joblib.dump(model, model_dir + "/titanic_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")

    # Specify the schema of the model's input/output (names, data types, ...)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Register the model in Hopsworks and upload it
    mr = project.get_model_registry()
    titanic_model = mr.python.create_model(
        name="titanic_modal",
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Survival Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    titanic_model.save(model_dir)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
