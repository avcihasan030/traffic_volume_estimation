from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from tqdm import tqdm
import time


# Load the training and test datasets
# Eğitim ve test veri setlerini yükleme
def load_data():
    train_data = pd.read_csv('data/processed_train_data.csv')
    test_data = pd.read_csv('data/processed_test_data.csv')
    return train_data, test_data


# Perform cross-validation on the model
# Model üzerinde cross-validation işlemini yapma
def perform_cross_validation(model, X_train, y_train):
    print("Cross-validation başlıyor...")

    # Cross-validation for Mean Squared Error (MSE)
    # Ortalama Kare Hatası (MSE) için cross-validation
    cv_scores_mse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_mse = -cv_scores_mse.mean()  # Negatif MSE değerlerini pozitif yapıyoruz
    print(f"Cross-validation MSE: {mean_cv_mse}")
    print(f"Cross-validation RMSE: {mean_cv_mse ** 0.5}")

    # Cross-validation for R^2 score
    # R^2 skoru için cross-validation
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_cv_r2 = cv_scores_r2.mean()
    print(f"Cross-validation R^2: {mean_cv_r2}")

    # Cross-validation for Mean Absolute Error (MAE)
    # Ortalama Mutlak Hata (MAE) için cross-validation
    cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -cv_scores_mae.mean()  # Negatif MAE değerlerini pozitif yapıyoruz
    print(f"Cross-validation MAE: {mean_cv_mae}")


# Build and train the RandomForest model
# RandomForest modelini oluşturma ve eğitme
def build_model():
    # Load the datasets
    # Veri setlerini yükleme
    train_data, test_data = load_data()

    # Separate features (X) and target variable (y)
    # Özellikler (X) ve hedef değişkeni (y) ayırma
    X = train_data.drop(columns=['traffic_volume'])
    y = train_data['traffic_volume']

    # Split the data into training and testing sets
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

    # Initialize the RandomForestRegressor model
    # RandomForestRegressor modelini başlatma
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform cross-validation on the training data
    # Eğitim verisi üzerinde cross-validation yapma
    perform_cross_validation(rf_model, X_train, y_train)

    # Train the model
    # Modeli eğitme
    print("Eğitim başlıyor...")
    rf_model.fit(X_train, y_train)

    # Make predictions on the test dataset
    # Test veri seti üzerinde tahmin yapma
    test_x = test_data.copy()
    if 'traffic_volume' in test_data.columns:
        test_x = test_x.drop(columns=['traffic_volume'])

    print("Tahmin yapılıyor...")
    test_predictions = rf_model.predict(test_x)

    # Check if the number of predictions matches the number of test samples
    # Tahminlerin sayısının test örneklerinin sayısıyla eşleşip eşleşmediğini kontrol etme
    if len(test_predictions) != len(test_data):
        raise ValueError('Number of predictions does not match number of test samples')

    # Add predictions to the test dataset and save to file
    # Tahminleri test veri setine ekleme ve dosyaya kaydetme
    test_data['prediction'] = test_predictions
    test_data.to_csv('test_predictions.csv', index=False)
    print("Predictions added to test dataset and saved to file.")

    # Display prediction results for both training and test sets
    # Eğitim ve test setleri için tahmin sonuçlarını gösterme
    display_prediction(rf_model, X_train, y_train, X_test, y_test)


# Display prediction results: RMSE, MAE, and R^2
# Tahmin sonuçlarını gösterme: RMSE, MAE ve R^2
def display_prediction(model, X_train, y_train, X_test, y_test):
    # Training data predictions and evaluation
    # Eğitim verisi tahminleri ve değerlendirmesi
    train_predictions = model.predict(X_train)
    print("-" * 30)
    print("Training RMSE:", mean_squared_error(y_train, train_predictions, squared=False))
    print("Training MAE:", mean_absolute_error(y_train, train_predictions))
    print("Training R^2:", r2_score(y_train, train_predictions))
    print("-" * 30)
    print("")
    print("-" * 30)
    # Test data predictions and evaluation
    # Test verisi tahminleri ve değerlendirmesi
    test_predictions = model.predict(X_test)
    print("Test RMSE:", mean_squared_error(y_test, test_predictions, squared=False))
    print("Test MAE:", mean_absolute_error(y_test, test_predictions))
    print("Test R^2:", r2_score(y_test, test_predictions))
    print("-" * 30)
