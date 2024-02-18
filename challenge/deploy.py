import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from model import DelayModel

data = pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)
model = DelayModel()

features, target = model.preprocess(
    data=data,
    target_column="delay"
)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

model.fit(
    features=features_train,
    target=target_train
)

target_predicted = model.predict(
    features=features_test
)

report = classification_report(target_test, target_predicted, output_dict=True)

with open("challenge/report.json", "w") as file:
    json.dump(report, file, indent=4)

model.save("challenge/delay_model.pkl")