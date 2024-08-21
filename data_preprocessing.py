import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
# Veriyi yükleme
def load_data(df):
    return df.copy()


# Handle missing values
# Eksik değerleri doldurma
def handle_null_values(df):
    data = df.copy()
    # 'is_holiday' sütunundaki eksik değerleri 'no' olarak doldur
    # Eğer değer 'no' değilse 'yes' yap
    data['is_holiday'] = data['is_holiday'].fillna('no').apply(lambda x: 'yes' if x != 'no' else 'no')
    return data


# Apply IQR method to remove outliers
# IQR yöntemiyle aykırı değerleri temizleme
def apply_iqr(df, iqr_columns):
    df_cleaned = df.copy()
    for column_name in iqr_columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Aykırı değerlerin alt ve üst sınırları dışında kalanları temizle
        df_cleaned = df_cleaned[(df_cleaned[column_name] >= lower_bound) & (df_cleaned[column_name] <= upper_bound)]
    return df_cleaned


# Apply Isolation Forest method to detect and remove outliers
# Isolation Forest yöntemiyle aykırı değerleri tespit etme ve temizleme
def apply_isolation_forest(df, if_columns, contamination=0.01):
    df_cleaned = df.copy()
    for column_name in if_columns:
        iso_forest = IsolationForest(contamination=contamination)
        df_cleaned['outlier'] = iso_forest.fit_predict(df_cleaned[[column_name]])
        # Aykırı değer olmayanları (outlier != -1) sakla ve diğerlerini temizle
        df_cleaned = df_cleaned[df_cleaned['outlier'] != -1].drop(columns=['outlier'])
    return df_cleaned


# Handle outliers using IQR and Isolation Forest methods
# IQR ve Isolation Forest yöntemlerini kullanarak aykırı değerleri temizleme
def handle_outliers(df, is_train):
    data = df.copy()

    iqr_columns = ['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction',
                   'visibility_in_miles', 'dew_point', 'clouds_all']

    # Eğitim verisinde 'traffic_volume' sütununu da IQR temizleme işlemlerine ekle
    if is_train:
        iqr_columns.append('traffic_volume')

    if_columns = ['temperature', 'rain_p_h', 'snow_p_h']

    # IQR yöntemiyle temizlenmiş veri seti
    df_cleaned_iqr = apply_iqr(data, iqr_columns)
    # Isolation Forest yöntemiyle temizlenmiş veri seti
    df_cleaned_isolation = apply_isolation_forest(data, if_columns)

    # Verileri dış birleşim yaparak birleştir (Fazla veri kaybını önlemek için)
    df_combined = pd.merge(df_cleaned_isolation, df_cleaned_iqr, how='outer')

    return df_combined


# Transform the dataset: extract time-related features
# Veri setini dönüştür: Zamanla ilgili özellikleri çıkarma
def transform_data(df):
    data = df.copy()

    data['date_time'] = pd.to_datetime(data['date_time'])

    # Zamanla ilgili özellikleri çıkarma
    data['year'] = data['date_time'].dt.year
    data['month'] = data['date_time'].dt.month
    data['day'] = data['date_time'].dt.day
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek

    # Orijinal date_time sütununu silme
    data.drop('date_time', axis=1, inplace=True)

    return data


# Encode categorical variables
# Kategorik değişkenleri encode etme
def encode_data(df):
    data = df.copy()

    # Kategorik değişkenleri encode etme (One-Hot Encoding)
    data = pd.get_dummies(data, columns=['is_holiday', 'weather_type'])

    # weather_description için Count Encoding
    count_encoding = data['weather_description'].value_counts().to_dict()
    data['weather_description_encoded'] = data['weather_description'].map(count_encoding)

    # Orijinal weather_description sütununu silme
    data.drop('weather_description', axis=1, inplace=True)

    return data


# Scale the data using Min-Max scaling
# Veriyi Min-Max ölçeklendirme ile normalize etme
def scale_data(df, is_train):
    data = df.copy()

    # Sayısal olmayan sütunları ayırma
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    numeric_data = data.drop(columns=non_numeric_columns)

    if is_train:
        # Hedef sütun olan 'traffic_volume'u ayır
        target_column = 'traffic_volume'
        features = numeric_data.drop(columns=[target_column])

        # MinMax normalizasyon işlemi
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        # Scaled veriyi tekrar DataFrame'e çevirme
        scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

        # Hedef sütunu tekrar ekleme
        scaled_df[target_column] = numeric_data[target_column].values

        # Sayısal olmayan sütunları tekrar ekleme
        scaled_df = pd.concat([scaled_df, data[non_numeric_columns].reset_index(drop=True)], axis=1)

        data = scaled_df.copy()

    else:
        # Eğitim dışı veri için ölçeklendirme
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=numeric_data.index)

        # Sayısal olmayan sütunları tekrar ekleme
        df_scaled = pd.concat([df_scaled, data[non_numeric_columns].reset_index(drop=True)], axis=1)

        data = df_scaled.copy()

    return data


# Split and reorder columns, with target column at the end
# Sütunları ayır ve yeniden sırala, hedef sütun en sona yerleştirilir
def split_and_merge_training_data(df):
    data = df.copy()
    cols = [col for col in data.columns if col != 'traffic_volume'] + ['traffic_volume']
    data_merged = data[cols]
    return data_merged


# Add missing columns to the test dataset
# Test veri setine eksik sütunları ekleme
def add_missing_column_to_test_data(df):
    test_data = df.copy()
    # 'weather_type_Squall' sütunu test veri setine eklenir
    test_data['weather_type_Squall'] = False

    # Sütunları istenilen sıraya göre yeniden düzenleme
    columns_order = [
        'air_pollution_index', 'humidity', 'wind_speed', 'wind_direction',
        'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h',
        'snow_p_h', 'clouds_all', 'year', 'month', 'day', 'hour', 'day_of_week',
        'weather_description_encoded', 'is_holiday_no',
        'is_holiday_yes', 'weather_type_Clear', 'weather_type_Clouds',
        'weather_type_Drizzle', 'weather_type_Fog', 'weather_type_Haze',
        'weather_type_Mist', 'weather_type_Rain', 'weather_type_Smoke',
        'weather_type_Snow', 'weather_type_Squall', 'weather_type_Thunderstorm'
    ]

    # Sütunları yeniden sıraya koyma
    test_data = test_data[columns_order]

    return test_data


# Preprocess the dataset (for both training and test sets)
# Veri setini ön işleme (hem eğitim hem de test setleri için)
def preprocess_data(df, is_train):
    data = load_data(df)

    data_without_null = handle_null_values(data)  # Eksik değerleri doldurma
    data_without_outliers = handle_outliers(data_without_null, is_train)  # Aykırı değerleri temizleme
    data_transformed = transform_data(data_without_outliers)  # Zamanla ilgili özellikleri çıkarma
    data_encoded = encode_data(data_transformed)  # Kategorik değişkenleri encode etme
    data_scaled = scale_data(data_encoded, is_train)  # Veriyi ölçeklendirme

    if is_train:
        # Eğitim verisi için sütunları sırala ve yeniden düzenle
        data = split_and_merge_training_data(data_scaled)
    else:
        # Test verisi için eksik sütunları ekle
        data = data_scaled.copy()
        data = add_missing_column_to_test_data(data)

    return data
