import sys
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import Qt

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "Data", "diabetes_prediction_dataset.csv")
MODEL = os.path.join(BASE, "Data", "model.pkl")


# ── model ────────────────────────────────────────────────────────────────────

def train_and_save():
    df = pd.read_csv(DATA)

    gender_enc   = LabelEncoder().fit(df['gender'])
    smoking_enc  = LabelEncoder().fit(df['smoking_history'])
    df['gender']          = gender_enc.transform(df['gender'])
    df['smoking_history'] = smoking_enc.transform(df['smoking_history'])

    X = df.drop('diabetes', axis=1).values
    y = df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    clf = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    clf.fit(X_train_bal, y_train_bal)

    acc = (clf.predict(X_test) == y_test).mean()
    print(f"Model accuracy: {acc * 100:.1f}%")

    with open(MODEL, "wb") as f:
        pickle.dump((scaler, clf, gender_enc, smoking_enc), f)

    return scaler, clf, gender_enc, smoking_enc


def load_model():
    if not os.path.exists(MODEL):
        return train_and_save()
    with open(MODEL, "rb") as f:
        return pickle.load(f)


# ── UI ────────────────────────────────────────────────────────────────────────

STYLE = """
QWidget {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI';
    font-size: 13px;
}
QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #cba6f7;
    padding: 8px 0 14px 0;
}
QLabel#section {
    font-size: 11px;
    color: #6c7086;
    letter-spacing: 1px;
    padding-top: 8px;
}
QLineEdit, QComboBox {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 8px;
    color: #cdd6f4;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #cba6f7;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
}
QPushButton#predict {
    background: #cba6f7;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 9px 32px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#predict:hover { background: #b4befe; }
QPushButton#clear {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 9px 22px;
    font-size: 13px;
}
QPushButton#clear:hover { background: #45475a; }
QPushButton#insight {
    background: #313244;
    color: #89b4fa;
    border: 1px solid #89b4fa;
    border-radius: 6px;
    padding: 9px 22px;
    font-size: 13px;
}
QPushButton#insight:hover { background: #1e1e2e; }
QLabel#result_pos {
    font-size: 14px;
    font-weight: bold;
    color: #f38ba8;
    padding: 4px 0;
}
QLabel#result_neg {
    font-size: 14px;
    font-weight: bold;
    color: #a6e3a1;
    padding: 4px 0;
}
QLabel#prob {
    font-size: 12px;
    color: #89b4fa;
}
"""


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.scaler, self.model, self.gender_enc, self.smoking_enc = load_model()
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("Diabetes Prediction")
        self.setMinimumWidth(500)
        self.resize(520, 560)
        self.setStyleSheet(STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 16, 28, 22)
        root.setSpacing(6)

        title = QLabel("Diabetes Prediction")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(16)

        def label(text):
            l = QLabel(text)
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return l

        # gender
        grid.addWidget(label("Gender"), 0, 0)
        self.gender = QComboBox()
        self.gender.addItems(["Female", "Male", "Other"])
        grid.addWidget(self.gender, 0, 1)

        # age
        grid.addWidget(label("Age"), 1, 0)
        self.age = QLineEdit()
        self.age.setPlaceholderText("e.g. 45")
        grid.addWidget(self.age, 1, 1)

        # hypertension
        grid.addWidget(label("Hypertension"), 2, 0)
        self.hypertension = QComboBox()
        self.hypertension.addItems(["No", "Yes"])
        grid.addWidget(self.hypertension, 2, 1)

        # heart disease
        grid.addWidget(label("Heart Disease"), 3, 0)
        self.heart_disease = QComboBox()
        self.heart_disease.addItems(["No", "Yes"])
        grid.addWidget(self.heart_disease, 3, 1)

        # smoking history
        grid.addWidget(label("Smoking History"), 4, 0)
        self.smoking = QComboBox()
        self.smoking.addItems(["never", "No Info", "current", "former", "ever", "not current"])
        grid.addWidget(self.smoking, 4, 1)

        # bmi
        grid.addWidget(label("BMI"), 5, 0)
        self.bmi = QLineEdit()
        self.bmi.setPlaceholderText("e.g. 27.5  (normal: 18.5–24.9)")
        grid.addWidget(self.bmi, 5, 1)

        # HbA1c
        grid.addWidget(label("HbA1c Level"), 6, 0)
        self.hba1c = QLineEdit()
        self.hba1c.setPlaceholderText("e.g. 5.7  (normal: below 5.7)")
        grid.addWidget(self.hba1c, 6, 1)

        # blood glucose
        grid.addWidget(label("Blood Glucose Level"), 7, 0)
        self.glucose = QLineEdit()
        self.glucose.setPlaceholderText("e.g. 130  (normal: 70–99 mg/dL)")
        grid.addWidget(self.glucose, 7, 1)

        root.addLayout(grid)
        root.addSpacing(10)

        # buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("predict")
        self.predict_btn.clicked.connect(self.predict)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clear")
        self.clear_btn.clicked.connect(self.clear_fields)
        self.insight_btn = QPushButton("Feature Importance")
        self.insight_btn.setObjectName("insight")
        self.insight_btn.clicked.connect(self.show_feature_importance)
        btn_row.addStretch()
        btn_row.addWidget(self.predict_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.insight_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addSpacing(8)

        self.result_lbl = QLabel("")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setWordWrap(True)
        root.addWidget(self.result_lbl)

        self.prob_lbl = QLabel("")
        self.prob_lbl.setObjectName("prob")
        self.prob_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.prob_lbl)

    def predict(self):
        # validate numeric fields
        try:
            age     = float(self.age.text().strip())
            bmi     = float(self.bmi.text().strip())
            hba1c   = float(self.hba1c.text().strip())
            glucose = float(self.glucose.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Age, BMI, HbA1c and Blood Glucose must be numbers.")
            return

        if not all([self.age.text(), self.bmi.text(), self.hba1c.text(), self.glucose.text()]):
            QMessageBox.warning(self, "Missing Input", "Please fill all fields.")
            return

        gender_val   = self.gender_enc.transform([self.gender.currentText()])[0]
        smoking_val  = self.smoking_enc.transform([self.smoking.currentText()])[0]
        hypertension = 1 if self.hypertension.currentIndex() == 1 else 0
        heart        = 1 if self.heart_disease.currentIndex() == 1 else 0

        features = np.array([[gender_val, age, hypertension, heart,
                               smoking_val, bmi, hba1c, glucose]])
        scaled  = self.scaler.transform(features)
        pred    = self.model.predict(scaled)[0]
        prob    = self.model.predict_proba(scaled)[0][1] * 100

        if pred == 1:
            self.result_lbl.setObjectName("result_pos")
            self.result_lbl.setText("High risk of diabetes detected.")
        else:
            self.result_lbl.setObjectName("result_neg")
            self.result_lbl.setText("Low risk — no diabetes detected.")

        self.prob_lbl.setText(f"Confidence: {prob:.1f}% chance of diabetes")
        self.result_lbl.setStyle(self.result_lbl.style())

    def show_feature_importance(self):
        feature_names = [
            "Gender", "Age", "Hypertension", "Heart Disease",
            "Smoking History", "BMI", "HbA1c Level", "Blood Glucose"
        ]
        importance = self.model.feature_importances_
        indices = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        bars = ax.barh(
            [feature_names[i] for i in indices],
            importance[indices],
            color="#cba6f7", edgecolor="none"
        )

        # highlight top 2
        sorted_imp = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)
        top2 = {sorted_imp[0][0], sorted_imp[1][0]}
        for bar, idx in zip(bars, indices):
            if idx in top2:
                bar.set_color("#f38ba8")

        ax.set_xlabel("Importance Score", color="#cdd6f4")
        ax.set_title("What drives the prediction?", color="#cba6f7", fontsize=13, pad=12)
        ax.tick_params(colors="#cdd6f4")
        ax.spines[:].set_color("#45475a")
        ax.xaxis.label.set_color("#6c7086")

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va='center', color="#cdd6f4", fontsize=9)

        plt.tight_layout()
        plt.show()

    def clear_fields(self):
        self.age.clear()
        self.bmi.clear()
        self.hba1c.clear()
        self.glucose.clear()
        self.gender.setCurrentIndex(0)
        self.smoking.setCurrentIndex(0)
        self.hypertension.setCurrentIndex(0)
        self.heart_disease.setCurrentIndex(0)
        self.result_lbl.setText("")
        self.prob_lbl.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
