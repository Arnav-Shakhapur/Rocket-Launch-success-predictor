import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

launches = requests.get("https://api.spacexdata.com/v4/launches").json()
rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()
launchpads = requests.get("https://api.spacexdata.com/v4/launchpads").json()
payloads = requests.get("https://api.spacexdata.com/v4/payloads").json()

launch_df = pd.json_normalize(launches)
rocket_df = pd.DataFrame(rockets)[["id","name"]].rename(columns={"id":"rocket","name":"rocket_name"})
pad_df = pd.DataFrame(launchpads)[["id","name","region"]].rename(columns={"id":"launchpad","name":"pad_name","region":"pad_region"})
payload_df = pd.DataFrame(payloads)[["id","mass_kg","orbit"]].rename(columns={"id":"payload_id"})

launch_df = launch_df.merge(rocket_df, on="rocket", how="left")
launch_df = launch_df.merge(pad_df, on="launchpad", how="left")

def total_payload_mass(pl):
    masses = payload_df[payload_df["payload_id"].isin(pl)]["mass_kg"]
    return masses.sum()

def primary_orbit(pl):
    orbits = payload_df[payload_df["payload_id"].isin(pl)]["orbit"]
    return orbits.iloc[0] if len(orbits) > 0 else None

def reused_core_count(cores):
    return sum([c["reused"] for c in cores]) if isinstance(cores,list) else 0

launch_df["payload_mass"] = launch_df["payloads"].apply(total_payload_mass)
launch_df["orbit"] = launch_df["payloads"].apply(primary_orbit)
launch_df["reuse_count"] = launch_df["cores"].apply(reused_core_count)
launch_df["year"] = pd.to_datetime(launch_df["date_utc"]).dt.year

launch_df = launch_df[[
    "success",
    "payload_mass",
    "orbit",
    "reuse_count",
    "rocket_name",
    "pad_region",
    "year"
]]

launch_df = launch_df.dropna()
launch_df["success"] = launch_df["success"].astype(int)

df = pd.get_dummies(
    launch_df,
    columns=["orbit","rocket_name","pad_region"],
    drop_first=True
)

sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

X = df.drop("success", axis=1)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(random_state=42)
params = {
    "n_estimators":[100,200],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5]
}

grid = GridSearchCV(model, params, scoring="roc_auc", cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, proba))

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.show()

importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind="barh")
plt.show()
