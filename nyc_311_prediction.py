import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

df = pd.read_csv(
    "311_Service_Requests_from_2020_to_Present_20261031.csv",
    usecols=[
        "Unique Key",
        "Created Date",
        "Closed Date",
        "Agency",
        "Problem (formerly Complaint Type)",
        "Problem Detail (formerly Descriptor)",
        "Location Type",
        "Incident Zip",
        "City",
        "Borough",
        "Facility Type",
        "Address Type",
        "Open Data Channel Type"
    ],
    low_memory=False
)

df["Created Date"] = pd.to_datetime(
    df["Created Date"], format="mixed", errors="coerce"
)

df["Closed Date"] = pd.to_datetime(
    df["Closed Date"], format="mixed", errors="coerce"
)

df["Resolution_Hours"] = (
    df["Closed Date"] - df["Created Date"]
).dt.total_seconds() / 3600

df["Created_Month"] = df["Created Date"].dt.month
df["Created_Day"] = df["Created Date"].dt.day
df["Created_Weekday"] = df["Created Date"].dt.dayofweek
df["Created_Hour"] = df["Created Date"].dt.hour

df.to_csv(
    "nyc_311_reduced.csv.gz",
    index=False,
    compression="gzip"
)

print("Original selected data shape:", df.shape)
print("File created successfully")
print("Missing closure dates:", df["Closed Date"].isna().sum())
print("Missing resolution hours:", df["Resolution_Hours"].isna().sum())
print(df.head())


print("\n--- DATA QUALITY CHECK ---")

print("Duplicate records:", df.duplicated().sum())

print("Negative resolution times:",
      (df["Resolution_Hours"] < 0).sum())

print("\nResolution time statistics:")
print(df["Resolution_Hours"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nRequests with zero resolution time:",
      (df["Resolution_Hours"] == 0).sum())

print("\nMissing values:")
print(df.isnull().sum())


print("\n--- REQUEST ID VALIDATION ---")

print("Duplicate request IDs:",
      df["Unique Key"].duplicated().sum())

print("Missing request IDs:",
      df["Unique Key"].isna().sum())

print("\n--- INVALID RESOLUTION RECORDS ---")

invalid = df[df["Resolution_Hours"] < 0]

print(invalid[[
    "Unique Key",
    "Created Date",
    "Closed Date",
    "Resolution_Hours"
]].head(10))

print("\nTotal invalid records:", len(invalid))

print("\n--- EXTREME RESOLUTION TIMES ---")

print(
    df.nlargest(10, "Resolution_Hours")[
        ["Unique Key", "Created Date", "Closed Date",
         "Resolution_Hours", "Agency",
         "Problem (formerly Complaint Type)"]
    ]
)

print("\n--- TARGET CLASS DISTRIBUTION ---")

valid_df = df[
    df["Created Date"].notna() &
    df["Closed Date"].notna() &
    (df["Resolution_Hours"] >= 0)
].copy()

valid_df["Resolution_Category"] = pd.cut(
    valid_df["Resolution_Hours"],
    bins=[-0.000001, 24, 168, float("inf")],
    labels=["Fast", "Medium", "Slow"]
)

print("Valid records:", len(valid_df))

print("\nClass counts:")
print(valid_df["Resolution_Category"].value_counts().sort_index())

print("\nClass percentages:")
print(
    (valid_df["Resolution_Category"]
     .value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

print("\nFinal target missing values:",
      valid_df["Resolution_Category"].isna().sum())
print("\n--- DATE RANGE VALIDATION ---")

print("Earliest request:",
      valid_df["Created Date"].min())

print("Latest request:",
      valid_df["Created Date"].max())

print("\nRecords by month:")

print(
    valid_df.groupby(
        valid_df["Created Date"].dt.to_period("M")
    ).size()
)
from sklearn.model_selection import train_test_split


sample_df = valid_df.sample(
    n=100000,
    random_state=42
).copy()

sample_df = sample_df.sort_values(
    "Created Date"
).reset_index(drop=True)

print("Sample shape:", sample_df.shape)

print("\nClass distribution:")
print(sample_df["Resolution_Category"].value_counts(normalize=True))

sample_df.to_csv(
    "nyc_311_sample.csv.gz",
    index=False,
    compression="gzip"
)


print("\n--- FINAL SAMPLE ---")
print("Sample shape:", sample_df.shape)

print("\nCategory distribution:")
print(sample_df["Resolution_Category"].value_counts())

print("\nCategory percentages:")
print(
    sample_df["Resolution_Category"]
    .value_counts(normalize=True) * 100
)

sample_df.to_csv(
    "nyc_311_sample.csv.gz",
    index=False,
    compression="gzip"
)
print("\n--- FINAL DATASET INSPECTION ---")

print("Shape:", sample_df.shape)

print("\nColumn names:")
print(sample_df.columns.tolist())

print("\nData types:")
print(sample_df.dtypes)

print("\nMissing values:")
print(sample_df.isnull().sum())

print("\nMemory usage:")
print(
    round(
        sample_df.memory_usage(deep=True).sum() / (1024 ** 2),
        2
    ),
    "MB"
)

print("\n--- FINAL MODEL PREPARATION ---")

model_df = sample_df.sort_values("Created Date").copy()

feature_cols = [
    "Agency",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Location Type",
    "Incident Zip",
    "Address Type",
    "City",
    "Borough",
    "Open Data Channel Type",
    "Created_Month",
    "Created_Day",
    "Created_Weekday",
    "Created_Hour"
]

X = model_df[feature_cols].copy()
y = model_df["Resolution_Category"].copy()

# Convert ZIP code into a categorical field
X["Incident Zip"] = (
    X["Incident Zip"]
    .round()
    .astype("Int64")
    .astype("string")
)

# Handle missing categorical values
categorical_cols = [
    "Agency",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Location Type",
    "Incident Zip",
    "Address Type",
    "City",
    "Borough",
    "Open Data Channel Type"
]

for col in categorical_cols:
    X[col] = X[col].fillna("Unknown")

# Chronological 80:20 split
split_index = int(len(model_df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining date range:")
print(model_df["Created Date"].iloc[0])
print(model_df["Created Date"].iloc[split_index - 1])

print("\nTesting date range:")
print(model_df["Created Date"].iloc[split_index])
print(model_df["Created Date"].iloc[-1])

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

print("\nMissing values in training features:")
print(X_train.isnull().sum().sum())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=sample_df,
    x="Resolution_Category",
    hue="Resolution_Category",
    palette="viridis",
    legend=False
)

plt.title("Distribution of NYC 311 Resolution Categories")
plt.xlabel("Resolution Category")
plt.ylabel("Number of Requests")
plt.tight_layout()
plt.show()

category_counts = sample_df["Resolution_Category"].value_counts()

plt.figure(figsize=(8, 5))

bars = plt.bar(
    category_counts.index,
    category_counts.values
)

plt.bar_label(bars, fmt="%d", padding=4)

plt.title("Distribution of NYC 311 Resolution Categories")
plt.xlabel("Resolution Category")
plt.ylabel("Number of Requests")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

plt.hist(
    sample_df["Resolution_Hours"],
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of NYC 311 Resolution Times")
plt.xlabel("Resolution Time (Hours)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

plt.hist(
    sample_df["Resolution_Hours"],
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of NYC 311 Resolution Times")
plt.xlabel("Resolution Time (Hours)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

plt.hist(
    sample_df["Resolution_Hours"],
    bins=50,
    range=(0, 500),
    edgecolor="black"
)

plt.title("NYC 311 Resolution Times (0–500 Hours)")
plt.xlabel("Resolution Time (Hours)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

top_complaints = sample_df[
    "Problem (formerly Complaint Type)"
].value_counts().head(10)

plt.figure(figsize=(11, 6))

plt.barh(
    top_complaints.index[::-1],
    top_complaints.values[::-1]
)

plt.title("Top 10 NYC 311 Complaint Types")
plt.xlabel("Number of Requests")
plt.ylabel("Complaint Type")

plt.tight_layout()
plt.show()

complaint_resolution = pd.crosstab(
    sample_df["Problem (formerly Complaint Type)"],
    sample_df["Resolution_Category"],
    normalize="index"
) * 100

top_10 = sample_df[
    "Problem (formerly Complaint Type)"
].value_counts().head(10).index

complaint_resolution = complaint_resolution.loc[top_10]

complaint_resolution.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Resolution Category by Top 10 Complaint Types")
plt.xlabel("Complaint Type")
plt.ylabel("Percentage of Requests")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Resolution Category")

plt.tight_layout()
plt.show()

borough_counts = sample_df["Borough"].value_counts()

plt.figure(figsize=(9, 5))

bars = plt.bar(
    borough_counts.index,
    borough_counts.values
)

plt.bar_label(bars, fmt="%d", padding=3)

plt.title("Distribution of NYC 311 Requests by Borough")
plt.xlabel("Borough")
plt.ylabel("Number of Requests")

plt.tight_layout()
plt.show()

borough_resolution = pd.crosstab(
    sample_df["Borough"],
    sample_df["Resolution_Category"],
    normalize="index"
) * 100

borough_resolution = borough_resolution.drop(
    index="Unspecified",
    errors="ignore"
)

borough_resolution.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Resolution Category by NYC Borough")
plt.xlabel("Borough")
plt.ylabel("Percentage of Requests")
plt.xticks(rotation=0)
plt.legend(title="Resolution Category")

plt.tight_layout()
plt.show()

borough_counts = pd.crosstab(
    sample_df["Borough"],
    sample_df["Resolution_Category"]
)

print("\n--- BOROUGH-WISE RESOLUTION COUNTS ---")
print(borough_counts)

numerical_cols = [
    "Resolution_Hours",
    "Created_Month",
    "Created_Day",
    "Created_Weekday",
    "Created_Hour"
]

correlation = sample_df[numerical_cols].corr()

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.show()

print("Missing values in training features:")
print(X_train.isnull().sum().sum())
categorical_cols = [
    "Agency",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Location Type",
    "Incident Zip",
    "Address Type",
    "City",
    "Borough",
    "Open Data Channel Type"
]

numerical_cols = [
    "Created_Month",
    "Created_Day",
    "Created_Weekday",
    "Created_Hour"
]
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20
    ))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_cols),
    ("numerical", numerical_pipeline, numerical_cols)
])
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\n--- PREPROCESSING RESULTS ---")

print("Original training shape:", X_train.shape)
print("Original testing shape:", X_test.shape)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

if hasattr(X_train_processed, "data"):
    print(
        "Missing values:",
        np.isnan(X_train_processed.data).any()
    )
else:
    print(
        "Missing values:",
        np.isnan(X_train_processed).any()
    )
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
}

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro F1": macro_f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))
    models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=1
    )
}
    print("\n--- MODELS REGISTERED ---")

for name in models:
    print(name)

    for name, model in models.items():

        print("\nTraining:", name)

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro F1": macro_f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))
print("\nTraining: Gaussian Naive Bayes")

X_train_dense = X_train_processed.astype("float32").toarray()
X_test_dense = X_test_processed.astype("float32").toarray()

gnb = GaussianNB()
gnb.fit(X_train_dense, y_train)

y_pred_gnb = gnb.predict(X_test_dense)

print("Accuracy:", round(accuracy_score(y_test, y_pred_gnb), 4))
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred_gnb), 4))
print("Macro F1:", round(f1_score(y_test, y_pred_gnb, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gnb, zero_division=0))

print("\nTraining: Support Vector Machine")

svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)

svm.fit(X_train_processed, y_train)

y_pred_svm = svm.predict(X_test_processed)

print("Accuracy:", round(accuracy_score(y_test, y_pred_svm), 4))
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred_svm), 4))
print("Macro F1:", round(f1_score(y_test, y_pred_svm, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=0))
    
