import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\SHIVAKUMAR\Desktop\PythonPrograms\internship\fraudTest.csv")
print("Original data shape:", df.shape)

# Step 1: Handle missing values
print("Missing values per column:\n", df.isnull().sum())
df.dropna(inplace=True)  # For simplicity; consider imputation for production

# Step 2: Identify categorical columns
object_cols = df.select_dtypes(include='object').columns
unique_counts = df[object_cols].nunique()

# Split by cardinality
low_card_cols = [col for col in object_cols if unique_counts[col] < 100]
high_card_cols = [col for col in object_cols if unique_counts[col] >= 100]

# One-hot encode low-cardinality columns
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

# Label encode high-cardinality columns
label_encoders = {}
for col in high_card_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 3: Feature scaling
scaler = StandardScaler()
if 'amount' in df.columns:
    df['amount'] = scaler.fit_transform(df[['amount']])

# Step 4: Prepare training data
y = df['is_fraud'] if 'is_fraud' in df.columns else df.iloc[:, -1]  # Assuming 'is_fraud' is the target
X = df.drop(columns=['is_fraud']) if 'is_fraud' in df.columns else df.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=importances[:15], y=importances.index[:15])
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# Optional: Identify misclassified examples
misclassified = X_test[y_test != y_pred]
print("\nSample Misclassifications:\n", misclassified.head())

