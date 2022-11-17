import os
import modal
    
BACKFILL=False
LOCAL=True

# NOTE: Change API keys here
# os.environ["HOPSWORKS_API_KEY"] 
#modal_secret_name = "hopsworks" # alternatives: "hopsworks" "HOPSWORKS_API_KEY"

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(modal_secret_name))
   def f():
       g()

def generate_passenger(survived, pclass=[1,2,3], age=[0,1]):
    """
    Returns a single titanic passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [float(random.choice(pclass))],
                       "sex": [float(random.randint(0, 1))],
                       "age": [float(random.choice(age))],
                      })
    df['Survived'] = survived
    return df

def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random titanic passenger
    """
    import random

    # create a survivor with class 1-2 and age Child-Teenager
    survived_df = generate_passenger(1, pclass=[1,2], age=[1,2])
    # create a non-survivor with class 2-3 and age Young Adult - Senior
    died_df = generate_passenger(0, pclass=[2,3], age=[3,4,5])

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survived_df
        print("Survived added")
    else:
        passenger_df = died_df
        print("Deceased added")

    return passenger_df

def fetch_and_preprocess_data():
    import pandas as pd

    df_titanic = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
    )

    # Drop columns with low predictive power
    df_titanic = df_titanic.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])

    # Drop more columns that won't make an intuitive UI
    # We assume Fare and passenger class correlate and it is more intuitive to use the passenger class
    df_titanic = df_titanic.drop(columns=["SibSp", "Parch", "Embarked", "Fare"])

    # Encode sex
    df_titanic["Sex"] = df_titanic["Sex"].apply(lambda x: 0 if x == "male" else 1)

    # Encode age by grouping to categories
    # Fill missing values with "unknown"
    df_titanic["Age"] = df_titanic["Age"].fillna(-1)

    # Create mapping, e.g. "Child" -> 1, "Teenager" -> 2, etc...
    age_labels = ["Unknown", "Child", "Teenager", "Young Adult", "Adult", "Senior"]
    age_mapping = dict([(age_labels[i], i) for i in range(0, len(age_labels))])

    df_titanic["Survived"] = df_titanic["Survived"].astype(int)
    df_titanic["Pclass"] = df_titanic["Pclass"].astype(float)
    df_titanic["Sex"] = df_titanic["Sex"].astype(float)
    
    
    # Aggregate ages into categories / bins
    df_titanic["Age"] = pd.cut(
        df_titanic["Age"],
        bins=[-2, -1, 12, 18, 35, 60, 120],
        labels=age_labels,
    )
    df_titanic["Age"] = df_titanic["Age"].apply(lambda a: age_mapping[a])

    df_titanic["Age"] = df_titanic["Age"].astype(float)

    print(df_titanic.dtypes)
    # Note: without astype int, the age column will be of type category
    # and become string on hopsworks
    return df_titanic


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = fetch_and_preprocess_data()
    else:
        titanic_df = get_random_titanic_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["Survived","Pclass","Sex","Age"], 
        description="Titanic survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()