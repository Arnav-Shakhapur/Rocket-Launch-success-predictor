import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "data" / "spacex_launches.json"
FIGURES_DIR = BASE_DIR / "figures"

API_URL = "https://ll.thespacedevs.com/2.2.0/launch/previous/"
PAGE_SIZE = 100

STATUS_SUCCESS = 3
STATUS_FAILURE = 4
STATUS_PARTIAL_FAILURE = 7


def get_with_retry(url, params=None, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.RequestException as err:
            wait = 2 ** attempt
            print(f"Network error ({err}); retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 60 * (attempt + 1)))
            print(f"Rate limited by Launch Library 2; waiting {wait}s")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            wait = 2 ** attempt
            print(f"Server error {response.status_code}; retrying in {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError(f"Could not fetch {url} after {max_attempts} attempts")


def fetch_spacex_launches():
    launches = []
    url = API_URL
    params = {"lsp__name": "SpaceX", "mode": "detailed", "limit": PAGE_SIZE, "ordering": "net"}

    while url:
        page = get_with_retry(url, params=params)
        launches.extend(page.get("results", []))
        print(f"Fetched {len(launches)} of {page.get('count', '?')} launches")
        url = page.get("next")
        params = None

    return launches


def load_launches():
    if CACHE_FILE.exists():
        print(f"Loading saved launch data from {CACHE_FILE.relative_to(BASE_DIR)}")
        return json.loads(CACHE_FILE.read_text())

    print("No saved data found; downloading from Launch Library 2")
    launches = fetch_spacex_launches()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(launches))
    return launches


def dig(record, *keys):
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def launch_to_row(launch):
    status_id = dig(launch, "status", "id")
    if status_id == STATUS_SUCCESS:
        success = 1
    elif status_id in (STATUS_FAILURE, STATUS_PARTIAL_FAILURE):
        success = 0
    else:
        return None

    stages = dig(launch, "rocket", "launcher_stage") or []
    reuse_count = sum(1 for stage in stages if stage.get("reused"))
    flight_numbers = [stage.get("launcher_flight_number") or 1 for stage in stages]

    return {
        "success": success,
        "year": pd.to_datetime(launch.get("net")).year,
        "reuse_count": reuse_count,
        "booster_flight_number": max(flight_numbers) if flight_numbers else 1,
        "rocket_name": dig(launch, "rocket", "configuration", "name"),
        "orbit": dig(launch, "mission", "orbit", "abbrev"),
        "mission_type": dig(launch, "mission", "type"),
        "pad_region": dig(launch, "pad", "location", "name"),
    }


def build_dataset(launches):
    rows = [row for row in map(launch_to_row, launches) if row is not None]
    df = pd.DataFrame(rows)


    categorical = ["rocket_name", "orbit", "mission_type", "pad_region"]
    df[categorical] = df[categorical].fillna("Unknown")

    return df, categorical


def save_figure(name):
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()


def main():
    launches = load_launches()
    launch_df, categorical = build_dataset(launches)

    print(f"\nLaunches with a final outcome: {len(launch_df)}")
    print("Class balance (1 = success, 0 = failure):")
    print(launch_df["success"].value_counts().to_string())

    df = pd.get_dummies(launch_df, columns=categorical, drop_first=True, dtype=int)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    plt.title("Feature correlations")
    save_figure("correlation_heatmap.png")

    X = df.drop("success", axis=1)
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    failures_in_train = int((y_train == 0).sum())
    n_splits = min(5, failures_in_train)
    if n_splits < 2:
        raise SystemExit("Not enough failed launches in the training data to cross validate.")

    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(model, params, scoring="roc_auc", cv=cv)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Cross validated ROC AUC ({n_splits} fold, training set): {grid.best_score_:.3f}")
    print(f"Failures in test set: {int((y_test == 0).sum())} of {len(y_test)}")
    print(f"Test set ROC AUC: {roc_auc_score(y_test, proba):.3f}")
    print("\n" + classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Failure", "Success"], yticklabels=["Failure", "Success"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test set)")
    save_figure("confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)")
    plt.legend()
    save_figure("roc_curve.png")

    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances.sort_values().tail(10).plot(kind="barh")
    plt.title("Top 10 drivers of launch success")
    save_figure("feature_importance.png")

    print(f"Charts saved to {FIGURES_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
