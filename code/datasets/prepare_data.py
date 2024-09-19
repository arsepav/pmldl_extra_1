import pandas as pd
from sklearn.model_selection import train_test_split

cat_features=['make', 'body', 'transmission', 'state', 'color', 'interior']
drop_features=['trim', 'model', 'vin', 'seller']
numeric_features=['year', 'condition', 'odometer', 'mmr', 'saledate']


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def clean_data(df):
    df = df.dropna(subset=['sellingprice'])

    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)

    df.loc[:, 'saledate'] = df['saledate'].apply(lambda x: x.timestamp() // 3600 if pd.notnull(x) else None)

    mean_time_in_hours = df['saledate'].mean()

    df['saledate'] = df['saledate'].fillna(mean_time_in_hours)

    for feature in cat_features:
        df.loc[:, feature] = df[feature].fillna('no')

    for feature in numeric_features:
        mean_value = df[feature].mean()
        df.loc[:, feature] = df[feature].fillna(mean_value)

    df = df.drop(columns=drop_features)

    return df


def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data


def save_data(train_data, test_data, train_path, test_path):
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


if __name__ == "__main__":
    raw_data_path = "data/raw/car_prices.csv"
    data = load_data(raw_data_path)
    
    cleaned_data = clean_data(data)
    
    train_data, test_data = split_data(cleaned_data)
    
    train_data_path = "data/processed/train_data.csv"
    test_data_path = "data/processed/test_data.csv"
    save_data(train_data, test_data, train_data_path, test_data_path)
