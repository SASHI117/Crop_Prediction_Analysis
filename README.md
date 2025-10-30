# 🌾 Crop Recommendation & Yield Prediction using XGBoost + SHAP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)]()
[![SHAP](https://img.shields.io/badge/ExplainableAI-SHAP-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

### 👤 Author: **Sashi Vardhan Pragada**

---

## 🧠 Project Overview
This project focuses on predicting **synthetic crop yield** using environmental and soil parameters like temperature, humidity, rainfall, and nutrient levels.  
It uses **XGBoost Regressor** for modeling and **SHAP (SHapley Additive Explanations)** for understanding feature importance.

🌱 The goal is to create an **accurate, interpretable ML model** that explains how each factor affects yield prediction — helping in better crop planning and soil management.

---

## 📂 Folder Structure
Crop-Recommendation/
├── Crop_recommendation.ipynb # Google Colab Notebook
├── Crop_recommendation.csv # Dataset
├── README.md # Documentation
└── requirements.txt # Dependencies

---

## 📊 1. Dataset Overview
**Dataset:** `Crop_recommendation.csv`  

| Feature | Description |
|----------|-------------|
| N | Nitrogen in soil |
| P | Phosphorous |
| K | Potassium |
| temperature | Temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH |
| rainfall | Rainfall (mm) |
| label | Crop type |

🧾 Renamed for clarity:
N → Nitrogen
P → Phosphorous
K → Potassium
temperature → Temparature
humidity → Humidity
rainfall → Rainfall
ph → pH
label → Crop_Type

---

## ⚙️ 2. Feature Engineering
Several derived features were created to enhance model performance and realism:

| Feature | Formula | Description |
|----------|----------|-------------|
| **NPK_Avg** | (N + P + K)/3 | Average nutrient concentration |
| **Total_Nutrients** | N + P + K | Overall soil fertility |
| **Temp_Humidity_Index** | (Temperature × Humidity)/100 | Combined effect of heat and moisture |
| **Moisture_norm** | (Rainfall − min)/(max − min) | Normalized rainfall |
| **Fertility_Level** | Quantile-based (Low/Medium/High) | Based on Total Nutrients |
| **Soil_cover_condition** | 0.6×Humidity + 0.4×Rainfall | Cover density of soil |

---

## 🌾 3. Synthetic Yield Target

Since actual yield data was unavailable, a synthetic target was generated:

\[
\text{synthetic\_yield\_target} = 0.5 × \text{NPK\_Avg} + 0.2 × \text{Humidity} + 0.1 × \text{Rainfall} + 5 × \text{temp\_factor}
\]

where  
\[
\text{temp\_factor} = \max(0, 1 - \frac{|\text{Temparature} - 27|}{28})
\]

🌡️ **27°C** is considered the ideal crop growth temperature.

---

## 🤖 4. Model Development

### Model Used
- **Algorithm:** XGBoost Regressor  
- **Goal:** Predict synthetic yield based on engineered features.

### Preprocessing
- **StandardScaler** → Scales numerical columns  
- **OneHotEncoder** → Encodes categorical variable (`Crop_Type`)  
- Combined using **ColumnTransformer** inside a **Pipeline**

### Data Split
- 80% Training  
- 20% Testing

### Hyperparameter Tuning (GridSearchCV)
| Parameter | Values Tested |
|------------|----------------|
| n_estimators | [100, 200] |
| learning_rate | [0.05, 0.1] |
| max_depth | [3, 5] |

✅ **Best parameters:**
{'xgb__learning_rate': 0.1, 'xgb__max_depth': 5, 'xgb__n_estimators': 200}

---

## 📈 5. Model Evaluation
Evaluation metrics used:
- **R² (Coefficient of Determination)**
- **Mean Squared Error (MSE)**

Example output:
R² = 0.9976
MSE = 0.5754

✅ The model performs excellently, explaining nearly all yield variance.

---

## 🔍 6. Explainable AI (SHAP Analysis)

### 📘 What is SHAP?
**SHAP (SHapley Additive Explanations)** assigns each feature a contribution value towards the prediction, derived from cooperative game theory.

### Formula:
\[
ϕ_i = \sum_{S⊆N\setminus{i}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S∪{i}) - f(S)]
\]

Where:
- \(ϕ_i\): SHAP value of feature *i*  
- \(f(S)\): model output for subset *S*

### 🧩 Interpretation
- **Positive SHAP value** → increases predicted yield  
- **Negative SHAP value** → decreases yield  
- **Summary plot** → visualizes impact of all features  
- **Bar plot** → ranks features by average importance

Top influencing features:
1. **NPK_Avg**
2. **Humidity**
3. **Total_Nutrients**

---

## 💡 7. Key Insights
- Nutrient levels (NPK) have the strongest effect on yield.  
- Humidity and rainfall also play important roles.  
- Optimal temperature (~27°C) gives the best yield.  
- SHAP confirms the model’s decisions are interpretable and realistic.

---

## 📄 8. Final Output
The model generates:
**`Crop_recommendation_with_predictions.csv`** containing:
- Original + derived features  
- Predicted yield (`Expected_yield`)  
- Fertility and soil cover condition  

---

## 🎯 9. Conclusion
This project demonstrates:
✅ Data preprocessing and feature creation  
✅ XGBoost regression with hyperparameter tuning  
✅ Explainability through SHAP  

📌 It provides a strong foundation for:
- Yield estimation  
- Fertilizer recommendation systems  
- Smart farming analytics  

🌍 **Future Work:**
- Integrate real yield data  
- Use IoT/remote sensing inputs  
- Build a live web dashboard for farmers  

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib  
- **Platform:** Google Colab  

---

## 🪄 How to Run
1. Open the notebook in **Google Colab**
2. Upload `Crop_recommendation.csv` when prompted
3. Run all cells sequentially
4. View:
   - Model performance metrics  
   - SHAP feature importance plots  
   - Final CSV output with predictions  

---

## 🧾 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
Contents of requirements.txt:
pandas
numpy
xgboost==2.0.3
shap==0.43.0
scikit-learn==1.3.2
matplotlib

---

## 🧑‍💻 Author

👋 Sashi Vardhan Pragada
Passionate about Artificial Intelligence, Machine Learning, and building explainable, data-driven solutions for real-world problems.

---

## 📜 License

Distributed under the MIT License — free to use, modify, and share with attribution.

---

## ⭐ If you found this project useful, consider giving it a star on GitHub! ⭐
