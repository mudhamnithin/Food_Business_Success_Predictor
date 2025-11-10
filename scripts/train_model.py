import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ============================================
# 1️⃣ Load dataset
# ============================================
df = pd.read_csv("data/hyderabad_food_outlets_with_features.csv")

feature_cols = [
    "footfall_index",
    "nearby_shops_count",
    "nearby_offices_count",
    "nearby_colleges_count",
    "nearby_hospitals_count",
    "nearby_restaurants_count",
    "nearby_parks_count",
    "distance_to_nearest_brand_chai",
    "rent_estimate",
    "avg_income_area"
]

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

# ============================================
# 2️⃣ Generate success label fairly
# ============================================
base_score = (
    df["footfall_index"] * 0.4 +
    df["avg_income_area"] * 0.3 +
    df["rent_estimate"] * 0.15 +
    df["nearby_offices_count"] * 0.1 +
    df["nearby_colleges_count"] * 0.05 -
    df["nearby_shops_count"] * 0.25
)

# Normalize score between 0 and 1
df["success_score_norm"] = (base_score - base_score.min()) / (base_score.max() - base_score.min())

# Add random noise to simulate uncertainty
rng = np.random.default_rng(42)
df["success_score_norm"] += rng.normal(0, 0.05, len(df))
df["success_score_norm"] = df["success_score_norm"].clip(0, 1)

# ✅ Adjust threshold to get ~30% successful businesses
df["success_label"] = np.where(df["success_score_norm"] > 0.20, 1, 0)

# ============================================
# 3️⃣ Label balance check
# ============================================
label_counts = df["success_label"].value_counts()
success_ratio = (label_counts[1] / len(df)) * 100

print("\n📊 Label Distribution:")
print(label_counts)
print(f"\n➡️ Success Ratio: {success_ratio:.2f}% successful businesses")

if success_ratio < 20:
    print("⚠️ Too few successful samples (<20%). Consider lowering threshold (e.g. 0.20).")
elif success_ratio > 50:
    print("⚠️ Too many successful samples (>50%). Consider raising threshold (e.g. 0.30).")
else:
    print("✅ Label balance is good (20–40% successes).")

# ============================================
# 4️⃣ Scale & Split
# ============================================
X = df[feature_cols]
y = df["success_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)

# ============================================
# 5️⃣ Balance Classes Using SMOTE
# ============================================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE:", np.bincount(y_train_res))

# ============================================
# 6️⃣ Train Model (Balanced Random Forest)
# ============================================
model = RandomForestClassifier(
    n_estimators=350,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# ============================================
# 7️⃣ Evaluate Model
# ============================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Fair Model Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix — Food Business Success Predictor")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
# 8️⃣ Cross-validation (to ensure stability)
# ============================================
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\n📊 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================
# 9️⃣ Feature Importance
# ============================================
feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
plt.title("🔍 Feature Importance — Key Drivers of Success")
plt.show()

# ============================================
# 🔟 Save model & scaler
# ============================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/food_success_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Model and scaler saved successfully in /models folder.")
