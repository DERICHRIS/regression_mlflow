import pandas as pd
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
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error,r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run():
    model=RandomForestRegressor(n_estimators=150,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X_cleaned,y_cleaned,test_size=0.2,random_state=42)

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    sme=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    mlflow.log_param("model","Random forest2")
    mlflow.log_metric("mean square error",sme)
    mlflow.log_metric("r2",r2)

    mlflow.sklearn.log_model(model,"Randomforest2")

    print(f"sme :{sme}")
    print(f"r2 : {r2}")
    


