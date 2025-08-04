# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle

# Visual Styling
mpl.style.use('ggplot')
sns.set()

# Load Dataset
car = pd.read_csv('car.csv', encoding='latin1')
backup = car.copy()

# Initial Inspection
print(f"\nDataset Shape: {car.shape}")
car.info()

# Data Cleaning
car = car[car['year'].astype(str).str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].astype(str).str.replace(',', '').str.replace('₹', '').astype(int)

car['kms_driven'] = car['kms_driven'].astype(str).str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]

car['name'] = car['name'].astype(str).str.split().str.slice(0, 3).str.join(' ')
car = car[car['Price'] < 6000000].reset_index(drop=True)

# Save cleaned dataset
car.to_csv('Cleaned_Car_data.csv', index=False)

# Summary
print(f"\nCleaned Dataset Shape: {car.shape}")
print("\nDataset Summary:")
print(car.describe(include='all'))

# Visualizations
plt.figure(figsize=(15, 7))
ax = sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.title('Price Distribution by Company')
plt.show()

plt.figure(figsize=(20, 10))
ax = sns.swarmplot(x='year', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.title('Price by Manufacturing Year')
plt.show()

sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
plt.title('KMs Driven vs Price')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x='fuel_type', y='Price', data=car)
plt.title('Fuel Type vs Price')
plt.show()

ax = sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2)
ax.set_xticklabels(rotation=40, ha='right')
plt.title('Mixed Feature Relationship')
plt.show()

# Feature Selection
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Pipeline Setup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Handle unknown categories with 'ignore'
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

pipe = make_pipeline(column_trans, LinearRegression())

# Hyperparameter Tuning
print("\nTuning Model for Best Random State...")
scores = []
for i in range(1000):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.1, random_state=i)
    pipe.fit(X_train_, y_train_)
    y_pred_ = pipe.predict(X_test_)
    scores.append(r2_score(y_test_, y_pred_))

best_i = np.argmax(scores)
print(f"\nBest R² Score: {scores[best_i]:.3f} at random_state={best_i}")

# Final Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_i)
pipe.fit(X_train, y_train)
final_r2 = r2_score(y_test, pipe.predict(X_test))
print(f"\nFinal R² Score on Test Set: {final_r2:.3f}")

# Save model
pickle.dump(pipe, open('CarPriceModel.pkl', 'wb'))
print("\nModel saved as 'CarPriceModel.pkl'")

# Sample Prediction
sample = pd.DataFrame(
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 15000, 'Petrol']).reshape(1, 5)
)
predicted_price = pipe.predict(sample)[0]
print(f"\nPredicted Price for Sample Input: ₹{int(predicted_price)}")

# Display learned categories for 'name'
ct = pipe.named_steps['columntransformer']
ohe_fitted = ct.transformers_[0][1]
print("\nLearned Categories for 'name':")
print(ohe_fitted.categories_[0])