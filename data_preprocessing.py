import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def load_data(df):
    return df.copy()


def handle_null_values(df):
    data = df.copy()
    data['is_holiday'] = data['is_holiday'].fillna('no').apply(lambda x: 'yes' if x != 'no' else 'no')
    return data


def apply_iqr(df, iqr_columns):
    df_cleaned = df.copy()
    for column_name in iqr_columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_cleaned = df_cleaned[(df_cleaned[column_name] >= lower_bound) & (df_cleaned[column_name] <= upper_bound)]
    return df_cleaned


def apply_isolation_forest(df, if_columns, contamination=0.01):
    df_cleaned = df.copy()
    for column_name in if_columns:
        iso_forest = IsolationForest(contamination=contamination)
        df_cleaned['outlier'] = iso_forest.fit_predict(df_cleaned[[column_name]])
        df_cleaned = df_cleaned[df_cleaned['outlier'] != -1].drop(columns=['outlier'])
    return df_cleaned


def handle_outliers(df, is_train):
    data = df.copy()

    iqr_columns = ['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction',
                   'visibility_in_miles', 'dew_point', 'clouds_all']

    if is_train:
        iqr_columns.append('traffic_volume')

    if_columns = ['temperature', 'rain_p_h', 'snow_p_h']

    df_cleaned_iqr = apply_iqr(data, iqr_columns)
    df_cleaned_isolation = apply_isolation_forest(data, if_columns)

    # Outer merge to avoid losing too much data
    df_combined = pd.merge(df_cleaned_isolation, df_cleaned_iqr, how='outer')

    return df_combined


def transform_data(df):
    data = df.copy()

    data['date_time'] = pd.to_datetime(data['date_time'])

    # Extracting time-related features
    data['year'] = data['date_time'].dt.year
    data['month'] = data['date_time'].dt.month
    data['day'] = data['date_time'].dt.day
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek

    # Drop the original date_time column
    data.drop('date_time', axis=1, inplace=True)

    return data


def encode_data(df):
    data = df.copy()

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['is_holiday', 'weather_type'])

    # Count encoding for weather_description
    count_encoding = data['weather_description'].value_counts().to_dict()
    data['weather_description_encoded'] = data['weather_description'].map(count_encoding)

    data.drop('weather_description', axis=1, inplace=True)

    return data


# def scale_data(df, is_train):
#     data = df.copy()
#
#     # Sayısal olmayan sütunları (örneğin, datetime sütunları) ayırın
#     non_numeric_columns = data.select_dtypes(exclude=['number']).columns
#     numeric_data = data.drop(columns=non_numeric_columns)
#
#     if is_train:
#         # Hedef sütunu ayır
#         target_column = 'traffic_volume'
#         features = numeric_data.drop(columns=[target_column])
#
#         # Standardizasyon işlemi
#         scaler = StandardScaler()
#         scaled_features = scaler.fit_transform(features)
#
#         # Scaled veriyi yeniden DataFrame'e çevir
#         scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
#
#         # Hedef sütunu geri ekle
#         scaled_df[target_column] = numeric_data[target_column].values
#
#         # Sayısal olmayan sütunları tekrar ekleyin
#         scaled_df = pd.concat([scaled_df, data[non_numeric_columns].reset_index(drop=True)], axis=1)
#
#         data = scaled_df.copy()
#
#     else:
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(numeric_data)
#         df_scaled = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=numeric_data.index)
#
#         # Sayısal olmayan sütunları tekrar ekleyin
#         df_scaled = pd.concat([df_scaled, data[non_numeric_columns].reset_index(drop=True)], axis=1)
#
#         data = df_scaled.copy()
#
#     return data


def scale_data(df, is_train):
    data = df.copy()

    # Sayısal olmayan sütunları (örneğin, datetime sütunları) ayırın
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    numeric_data = data.drop(columns=non_numeric_columns)

    if is_train:
        # Hedef sütunu ayır
        target_column = 'traffic_volume'
        features = numeric_data.drop(columns=[target_column])

        # MinMax normalizasyon işlemi
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        # Scaled veriyi yeniden DataFrame'e çevir
        scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

        # Hedef sütunu geri ekle
        scaled_df[target_column] = numeric_data[target_column].values

        # Sayısal olmayan sütunları tekrar ekleyin
        scaled_df = pd.concat([scaled_df, data[non_numeric_columns].reset_index(drop=True)], axis=1)

        data = scaled_df.copy()

    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=numeric_data.index)

        # Sayısal olmayan sütunları tekrar ekleyin
        df_scaled = pd.concat([df_scaled, data[non_numeric_columns].reset_index(drop=True)], axis=1)

        data = df_scaled.copy()

    return data


def split_and_merge_training_data(df):
    data = df.copy()
    cols = [col for col in data.columns if col != 'traffic_volume'] + ['traffic_volume']
    data_merged = data[cols]
    return data_merged


def add_missing_column_to_test_data(df):
    test_data = df.copy()
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


def preprocess_data(df, is_train):
    data = load_data(df)

    data_without_null = handle_null_values(data)
    data_without_outliers = handle_outliers(data_without_null, is_train)
    data_transformed = transform_data(data_without_outliers)
    data_encoded = encode_data(data_transformed)
    data_scaled = scale_data(data_encoded, is_train)

    if is_train:
        data = split_and_merge_training_data(data_scaled)
    else:
        data = data_scaled.copy()
        data = add_missing_column_to_test_data(data)

    return data
