import os
import modal

BACKFILL=False
LOCAL=False

# NOTE: Change API keys here
#os.environ["HOPSWORKS_API_KEY"] = ""
modal_secret_name = "HOPSWORKS_API_KEY" # alternatives: "hopsworks" "HOPSWORKS_API_KEY"

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(modal_secret_name))
   def f():
       g()

def generate_passenger(survived, fare_max, fare_min, pclass=[1,2,3], age=[0,1]):
    """
    Returns a single titanic passenger as a single row in a DataFrame
    """

    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [random.choice(pclass)],
                       "sex": [random.randint(0, 1)],
                       "age": [random.choice(age)],
                       "fare": [random.randint(fare_min,fare_max)],
                      })

    df['survived'] = survived
    return df

def is_unique(df_full, df_passenger):
    """Checks if the generated passenger represents a unique data point"""
    df = df_full.drop_duplicates().append(df_passenger)
    return not df.duplicated().any()
    

def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random titanic passenger
    """
    import random
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df_full = titanic_fg.read()

    # randomly pick one of these 2 and write it to the featurestore if it is a unique datapoint
    max_tries = 100
    for i in range(max_tries):
        if random.uniform(0,2) >= 1:
            # create a survivor with class 2-3 and age Child-Teenager
            df_passenger = generate_passenger(1, pclass=[2,3], age=[1,2],fare_min=0, fare_max=4)
        else:
            # create a non-survivor with class 1 and age Adult - Senior
            df_passenger = generate_passenger(0, pclass=[1], age=[4,5],fare_min=5,fare_max=10)
        
        if is_unique(df_full, df_passenger):
            print(f"Return data point after {i} tries")
            print(df_passenger)
            return df_passenger.astype(int)

    print(f"Could not generate a unique datapoint after {max_tries} tries")


def fetch_and_preprocess_data():

    # %%
    import pandas as pd
    import seaborn as sb

    df_titanic = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
    )

    # Drop columns with low predictive power
    df_titanic = df_titanic.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])

    # Drop more columns that won't make an intuitive UI
    # We assume Fare and passenger class correlate and it is more intuitive to use the passenger class
    df_titanic = df_titanic.drop(columns=["SibSp", "Parch", "Embarked"])

    # Encode sex
    df_titanic["Sex"] = df_titanic["Sex"].apply(lambda x: 0 if x == "male" else 1)

    # Encode age by grouping to categories
    # Fill missing values with "unknown"
    df_titanic["Age"] = df_titanic["Age"].fillna(-1)


    # Create mapping, e.g. "Child" -> 1, "Teenager" -> 2, etc...
    age_labels = ["Unknown", "Child", "Teenager", "Young Adult", "Adult", "Senior"]
    age_mapping = dict([(age_labels[i], i) for i in range(0, len(age_labels))])

    # Aggregate ages into categories / bins
    df_titanic["Age"] = pd.cut(
        df_titanic["Age"],
        bins=[-2, -1, 12, 18, 35, 60, 120],
        labels=age_labels,
    )

    df_titanic["Age"] = df_titanic["Age"].apply(lambda a: age_mapping[a])

    fare_max = df_titanic["Fare"].max()
    fare_bins = [-1,10,25,50,75,100,125,150,200,250,300,fare_max]
    fare_bin_labels = [0,1,2,3,4,5,6,7,8,9,10]
    df_titanic["Fare"] = pd.cut(df_titanic['Fare'], fare_bins, labels = fare_bin_labels)

    # Many inputs appear multiple times, so we have to discard duplicates,
    # but keep the right label.
    # First, create a mapping of data samples to how often they occur
    cols = list(df_titanic.columns)
    label_freq = df_titanic.groupby(by=cols).size().reset_index().rename(columns={0: "count"})
    label_freq = label_freq.set_index(cols).to_dict()["count"]

    def get_label(row):
        # row is a tuple of (survived, pclass, sex, age, fare)
        inputs = list(row[1:])
        try:
            survived_count = label_freq[tuple([1] + inputs)]
        except:
            return 0
        try:
            died_count = label_freq[tuple([0] + inputs)]
        except:
            return 1
        return int(survived_count > died_count)

    # Drop duplicates (samples with the same input values)
    # and choose the label that occurs most often
    input_cols = cols[1:]
    df_titanic = df_titanic.drop_duplicates(subset=input_cols)
    df_titanic["Survived"] = df_titanic.apply(get_label, axis=1)
    df_titanic = df_titanic.reset_index(drop=True).astype(int)
    df_titanic.columns = df_titanic.columns.str.lower()

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
        primary_key=["Survived","Pclass","Sex","Age","Fare"], 
        description="Titanic survival dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
# %%
