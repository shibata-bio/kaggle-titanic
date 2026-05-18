import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

# 欠損値処理
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 性別を数値化
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# 特徴量
X = df[["Pclass", "Sex", "Age"]]

# 正解データ
y = df["Survived"]

# モデル作成
model = RandomForestClassifier()

# 学習
model.fit(X, y)

# 予測
predictions = model.predict(X)

print(predictions[:10])