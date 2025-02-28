import pandas as pd

# Read the data
data = pd.read_csv('C:/Users/deric/Desktop/mlflow_test/regression_algo/Life Expectancy Data.csv')

# Select relevant features and target
X = data[['Adult Mortality','infant deaths','Alcohol','percentage expenditure',' BMI ','under-five deaths ','GDP','Schooling','Polio']]  # Choose relevant features
y = data['Life expectancy ']  # Target variable

# Clean column names by stripping any extra spaces
X.columns = X.columns.str.strip()

# Remove rows with NaN values in any of the selected features (X)
X_cleaned = X.dropna()

# Remove corresponding rows in the target variable (y) to keep alignment
y_cleaned = y[X_cleaned.index]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://127.0.0.1:5000')
with mlflow.start_run():
    X_train,X_test,y_train,y_test=train_test_split(X_cleaned,y_cleaned,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the Neural Network model
    model = Sequential()

    # Input layer (input shape should match the number of features)
    model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))

    # Hidden layer
    model.add(Dense(64, activation='relu'))

    # Hidden layer2
    model.add(Dense(32, activation='relu'))

    # Output layer (single neuron for regression task)
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    sme=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    mlflow.log_param("model","ANN2")
    mlflow.log_metric("mean square error",sme)
    mlflow.log_metric("r2",r2)

    mlflow.sklearn.log_model(model,"ANN2")

    print(f"sme :{sme}")
    print(f"r2 : {r2}")

