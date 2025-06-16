import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os # <-- Tambahkan import os

# Aktifkan autologging
mlflow.sklearn.autolog()

# --- Ganti bagian pemuatan data ---

# Menentukan path secara dinamis
# Ini membuat skrip dapat menemukan file CSV tidak peduli dari mana ia dijalankan
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'heart-disease-uci_preprocessing.csv')
df = pd.read_csv(file_path)

# --- Akhir dari bagian penggantian ---


# Pisahkan fitur dan target
X = df.drop("num", axis=1)
y = df["num"]

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai MLflow run
with mlflow.start_run():
    # Inisialisasi dan training model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)