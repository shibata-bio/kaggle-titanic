import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent

# データ読み込み
df = pd.read_csv(BASE_DIR / "data" / "train.csv")

# 欠損値処理
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna("S")

# 文字を数値化
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 特徴量
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# モデル作成・学習
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 予測・評価
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance")

print(importance)

# グラフ化
plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.tight_layout()

# 保存
output_path = BASE_DIR / "images" / "feature_importance.png"
output_path.parent.mkdir(exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output_path)
