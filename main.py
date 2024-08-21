import pandas as pd

from data_modeling import build_model
from data_preprocessing import preprocess_data


# Veri Yükleme
def load_dataset(file_path):
    """
    CSV dosyasından veri yükler.
    """
    return pd.read_csv(file_path)


# Ana fonksiyon
def process():
    # Dosya yollarını belirt
    train_file_path = 'data/train_data.csv'  # Eğitilecek verilerin dosya yolu
    test_file_path = 'data/test_data.csv'  # Test edilecek verilerin dosya yolu

    # Verileri yükle
    print("Veriler yükleniyor...")
    train_data = load_dataset(train_file_path)
    test_data = load_dataset(test_file_path)

    # Veri ön işleme
    print("Eğitim verileri ön işleniyor...")
    processed_train_data = preprocess_data(train_data, is_train=True)

    print("Test verileri ön işleniyor...")
    processed_test_data = preprocess_data(test_data, is_train=False)

    # İşlenmiş verileri kaydet
    processed_train_file_path = 'data/processed_train_data.csv'
    processed_test_file_path = 'data/processed_test_data.csv'

    print(f"Eğitim verileri kaydediliyor: {processed_train_file_path}")
    processed_train_data.to_csv(processed_train_file_path, index=False)

    print(f"Test verileri kaydediliyor: {processed_test_file_path}")
    processed_test_data.to_csv(processed_test_file_path, index=False)

    print("Veri ön işleme tamamlandı.")


if __name__ == "__main__":
    # process()
    build_model()
