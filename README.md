#  Swiggy Delivery Time Prediction

A Machine Learning project that predicts **food delivery time in minutes** based on rider, order, location, traffic, weather, and other delivery-related factors.

The project includes **data preprocessing, exploratory data analysis, feature engineering, multiple regression models, hyperparameter tuning with Optuna, model evaluation, and a Streamlit web application** for making real-time predictions.

---

##  Project Overview

Food delivery time depends on several factors such as:

* Rider age and rating
* Traffic conditions
* Weather
* Distance between restaurant and customer
* Vehicle condition and type
* Number of multiple deliveries
* Order type
* Festival conditions
* City and city type
* Pickup time
* Order time and time of day

The goal of this project is to build a regression model that can estimate the expected delivery time for a given order.

---

## Objective

To develop a machine learning model that predicts **delivery time (`time_taken`) in minutes** and deploy the trained model through an interactive Streamlit application.

---

##  Dataset

The dataset contains **45,502 records and 26 columns**.

The target variable is:

```text
time_taken
```

The dataset contains rider information, geographical coordinates, order information, weather, traffic, city information, and timing-related features.

### Important Features

| Category          | Features                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| Rider             | `age`, `ratings`, `vehicle_condition`                                                                 |
| Location          | `restaurant_latitude`, `restaurant_longitude`, `delivery_latitude`, `delivery_longitude`              |
| Order             | `type_of_order`, `multiple_deliveries`, `festival`                                                    |
| Vehicle           | `type_of_vehicle`                                                                                     |
| Traffic & Weather | `traffic`, `weather`                                                                                  |
| City              | `city_name`, `city_type`                                                                              |
| Date & Time       | `order_day`, `order_month`, `order_day_of_week`, `is_weekend`, `order_time_hour`, `order_time_of_day` |
| Delivery          | `pickup_time_minutes`, `distance`                                                                     |
| Target            | `time_taken`                                                                                          |

---

##  Data Preprocessing

The dataset was inspected for:

* Shape and data types
* Missing values
* Duplicate records
* Unique values
* Descriptive statistics

The dataset initially contained missing values in several numerical and categorical columns. No duplicate rows were found.

### Preprocessing Pipeline

Different preprocessing techniques were applied based on feature type.

#### Numerical Features

* Median imputation
* Standard scaling

#### Ordinal Features

The following features were ordinal encoded:

* `traffic`
* `order_time_of_day`
* `city_type`

Unknown categories are handled using an encoded value of `-1`.

#### Categorical Features

One-hot encoding was applied to:

* `weather`
* `type_of_order`
* `type_of_vehicle`
* `festival`
* `city_name`
* `order_day_of_week`

Unknown categories are ignored during transformation.

---

##  Feature Engineering

Additional features were used to improve the prediction:

* Order day
* Order month
* Day of the week
* Weekend indicator
* Order hour
* Time-of-day category
* Distance between restaurant and delivery location

Distance is calculated using geographical coordinates and the **Haversine formula**.

The Streamlit application also automatically calculates the distance from restaurant and delivery coordinates.

---

##  Machine Learning Models

The following regression algorithms were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. KNN Regressor

### Model Performance

| Model             |      RMSE |       MAE |   R² Score |
| ----------------- | --------: | --------: | ---------: |
| **Random Forest** | **4.163** | **3.295** | **0.8030** |
| Gradient Boosting |     4.711 |     3.746 |     0.7476 |
| Decision Tree     |     5.668 |     4.309 |     0.6348 |
| Linear Regression |     6.163 |     4.917 |     0.5682 |
| KNN Regressor     |     6.434 |     5.039 |     0.5294 |

Random Forest performed best among the initial models, with an R² score of approximately **0.803**.

---

##  Hyperparameter Tuning

**Optuna** was used to tune the Random Forest model.

The parameters optimized were:

* `n_estimators`
* `max_depth`
* `min_samples_split`

### Best Parameters

```text
n_estimators = 210
max_depth = 13
min_samples_split = 8
```

The best cross-validation R² score during tuning was approximately **0.8014**.

---

##  Final Model

The final model is a **Tuned Random Forest Regressor** combined with the preprocessing pipeline.

### Final Test Performance

| Metric       |             Score |
| ------------ | ----------------: |
| **MSE**      |        **16.951** |
| **RMSE**     | **4.117 minutes** |
| **MAE**      | **3.264 minutes** |
| **R² Score** |        **0.8073** |

The model achieved an R² of **0.8073** on the unseen test set.

### Training vs Testing

```text
Training R² : 0.8582
Testing R²  : 0.8073
Gap         : 0.0510
```

The relatively small train-test gap indicates that the final model does not show a large amount of overfitting.

---

##  Important Features

The most influential features in the final Random Forest model included:

| Feature               | Importance |
| --------------------- | ---------: |
| `ratings`             |     0.2219 |
| `traffic`             |     0.1393 |
| `multiple_deliveries` |     0.1340 |
| `distance`            |     0.0943 |
| `age`                 |     0.0941 |
| `weather_sunny`       |     0.0807 |
| `vehicle_condition`   |     0.0773 |

Rider rating was the highest-ranked feature according to the model's feature importance values.

---

##  Streamlit Application

The trained model is deployed through a **Streamlit web application** called:

> **Swiggy Delivery Time Predictor**

The application accepts inputs such as:

* Rider age
* Rider rating
* Vehicle condition
* Vehicle type
* Multiple deliveries
* Order type
* Festival
* Restaurant coordinates
* Delivery coordinates
* City
* City type
* Weather
* Traffic
* Order date
* Order hour
* Pickup time

## The application then sends the processed input to the trained pipeline and displays the estimated delivery time in minutes.

## Project Structure

```text
Swiggy-Delivery-Time-Prediction/
│
├── Swiggy_Delivery_Time_Prediction_.ipynb
├── app.py
├── swiggy_pipeline.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                                     | Description                                                 |
| ---------------------------------------- | ----------------------------------------------------------- |
| `Swiggy_Delivery_Time_Prediction_.ipynb` | Data analysis, preprocessing, model training and evaluation |
| `app.py`                                 | Streamlit web application                                   |
| `swiggy_pipeline.pkl`                    | Saved preprocessing + trained Random Forest pipeline        |
| `requirements.txt`                       | Python dependencies                                         |
| `README.md`                              | Project documentation                                       |

The notebook saves the final trained pipeline as `swiggy_pipeline.pkl`.

---

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Optuna
* Streamlit
* Pickle

---

##  How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Swiggy-Delivery-Time-Prediction.git
cd Swiggy-Delivery-Time-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure the model file is present

Place:

```text
swiggy_pipeline.pkl
```

in the same directory as `app.py`.

The Streamlit application expects this saved pipeline to load the trained preprocessing and Random Forest model.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL in the terminal, usually:

```text
http://localhost:8501
```

---

##  Model Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Missing Value Analysis
     ↓
Duplicate Check
     ↓
Feature Engineering
     ↓
Train-Test Split
     ↓
Preprocessing Pipeline
     ↓
Multiple Regression Models
     ↓
Model Comparison
     ↓
5-Fold Cross Validation
     ↓
Optuna Hyperparameter Tuning
     ↓
Tuned Random Forest
     ↓
Model Evaluation
     ↓
Save Pipeline
     ↓
Streamlit Deployment
```

---

##  Prediction

The application produces an estimated delivery time such as:

```text
Estimated delivery time: 28.5 minutes
```

The prediction is generated directly from the saved trained pipeline.

---

##  Key Takeaways

* Built a complete end-to-end machine learning regression project.
* Processed a dataset containing **45,502 delivery records**.
* Compared five different regression algorithms.
* Random Forest performed best among the baseline models.
* Used **5-fold cross-validation** to evaluate model consistency.
* Used **Optuna** for Random Forest hyperparameter tuning.
* Final model achieved **0.8073 R²** on the test set.
* Deployed the trained model using **Streamlit**.
* Saved the complete preprocessing and model pipeline using Pickle.

---

## Author

**Tejaswi Siddagoni**

If you found this project useful, feel free to ⭐ the repository.
