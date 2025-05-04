import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

# Open a log file
log_file = open('car_price_prediction_output.log', 'w')

def log(message):
    """Print to console and write to log file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

log("Car Price Prediction Demo")
log("========================\n")

# Load the dataset
log("Loading dataset...")
try:
    # Ensure stdout is flushed immediately
    sys.stdout.flush()
    
    data = pd.read_csv("Data/data.csv")
    log(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.\n")
    sys.stdout.flush()
    
    # Display first few rows (just first 2 rows to keep it concise)
    log("First 2 rows of the dataset (sample):")
    log(str(data.head(2)))
    log("\n")
    sys.stdout.flush()

    # Data info
    log("Dataset information:")
    log(f"Number of missing values: {data.isnull().sum().sum()}")
    log(f"Number of car manufacturers: {data['Make'].nunique()}")
    log(f"Year range: {data['Year'].min()} - {data['Year'].max()}")
    log(f"Price range: ${int(data['MSRP'].min())} - ${int(data['MSRP'].max())}")
    log("\n")
    sys.stdout.flush()

    # Top car manufacturers
    log("Top 5 car manufacturers by count:")
    top_5_makes = data['Make'].value_counts().head(5)
    for make, count in top_5_makes.items():
        log(f"  {make}: {count} cars")
    log("\n")
    sys.stdout.flush()
    
    # Most expensive car manufacturers
    log("Top 5 most expensive car manufacturers (average price):")
    avg_price_by_make = data.groupby('Make')['MSRP'].mean().sort_values(ascending=False).head(5)
    for make, price in avg_price_by_make.items():
        log(f"  {make}: ${int(price)}")
    log("\n")
    sys.stdout.flush()

    # Basic visualization
    log("Creating visualizations...")
    sys.stdout.flush()
    
    # Skip visualizations for now to ensure the script runs quickly
    
    # Data preparation for modeling
    log("Preparing data for modeling...")
    sys.stdout.flush()
    
    # Handle missing values
    data = data.dropna()
    log(f"Data shape after dropping missing values: {data.shape}")
    sys.stdout.flush()
    
    # Select features and target
    features = ['Year', 'Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'Popularity']
    X = data[features]
    y = data['MSRP']
    
    log(f"Selected features: {', '.join(features)}")
    sys.stdout.flush()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")
    sys.stdout.flush()
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    log("Training Decision Tree Regressor model...")
    sys.stdout.flush()
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    log("Model training completed.")
    sys.stdout.flush()
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    log("\nModel Evaluation:")
    log(f"Mean Absolute Error: ${int(mae)}")
    log(f"Mean Squared Error: ${int(mse)}")
    log(f"Root Mean Squared Error: ${int(rmse)}")
    sys.stdout.flush()
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = {features[i]: importance[i] for i in range(len(features))}
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    
    log("\nFeature Importance:")
    for feature, importance in sorted_importance.items():
        log(f"  {feature}: {importance:.4f}")
    sys.stdout.flush()
    
    # Sample prediction
    log("\nSample predictions for new cars:")
    sys.stdout.flush()
    
    # Create a sample of 3 cars with different characteristics
    sample_cars = [
        {'Year': 2022, 'Engine HP': 300, 'Engine Cylinders': 6, 'highway MPG': 30, 'city mpg': 22, 'Popularity': 1000},
        {'Year': 2022, 'Engine HP': 200, 'Engine Cylinders': 4, 'highway MPG': 40, 'city mpg': 30, 'Popularity': 800},
        {'Year': 2022, 'Engine HP': 450, 'Engine Cylinders': 8, 'highway MPG': 25, 'city mpg': 18, 'Popularity': 1200}
    ]
    
    sample_df = pd.DataFrame(sample_cars)
    sample_scaled = scaler.transform(sample_df)
    predictions = model.predict(sample_scaled)
    
    for i, car in enumerate(sample_cars):
        log(f"Car {i+1}:")
        for k, v in car.items():
            log(f"  {k}: {v}")
        log(f"  Predicted Price: ${int(predictions[i])}\n")
    sys.stdout.flush()
    
    log("Car Price Prediction Demo completed successfully!")
    sys.stdout.flush()

except Exception as e:
    log(f"An error occurred: {e}")
    import traceback
    traceback.print_exc(file=log_file)
    sys.stdout.flush()

# Close the log file
log_file.close()

# Let the user know where to find the results
print(f"Script execution completed. Results saved to {os.path.abspath('car_price_prediction_output.log')}") 