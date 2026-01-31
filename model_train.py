import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('stock_data.csv')

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

X = df.drop('Stock_5', axis=1)
y = df['Stock_5']

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1    
)

rf_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', rf_model)   
])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


with open("stock_rf_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("Random Forest pipeline saved as stock_rf_pipeline.pkl")
