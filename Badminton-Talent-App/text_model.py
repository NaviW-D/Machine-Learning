"""
Text-Based Talent Prediction for Badminton
Author: Navid
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 1. بارگذاری داده از CSV
# -----------------------------
data = pd.read_csv("data/user_text_data.csv")

# -----------------------------
# 2. آماده‌سازی ویژگی‌ها و برچسب‌ها
# -----------------------------
# در این مرحله همه خروجی‌ها Badminton هستند (1)
data['recommended_sport'] = 1

X = data[['age', 'height_cm', 'weight_kg', 'experience_years', 'interest_level']]
y = data['recommended_sport']

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. ساخت مدل KNN
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# -----------------------------
# 4. پیش‌بینی رشته ورزشی برای نمونه جدید
# -----------------------------
# نمونه کاربر جدید
new_user = pd.DataFrame({
    'age':[15],
    'height_cm':[162],
    'weight_kg':[53],
    'experience_years':[1],
    'interest_level':[5]
})

new_user_scaled = scaler.transform(new_user)
prediction = knn.predict(new_user_scaled)

print("Predicted Sport for new user:", "Badminton" if prediction[0]==1 else "Other")
