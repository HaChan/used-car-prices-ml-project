import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Load the data
df_res = pd.read_csv("data/sample_submission.csv")
df_test = pd.read_csv("data/test.csv")
df_train = pd.read_csv("data/train.csv")

# Handle missing values
def convert_clean_title_to_bool_val(df):
    df['clean_title'] = df['clean_title'].map(lambda x: 1 if x == 'Yes' else 0)

convert_clean_title_to_bool_val(df_train)
convert_clean_title_to_bool_val(df_test)


def convert_accident_to_bool_val(df):
    df['accident'] = df['accident'].map(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

convert_accident_to_bool_val(df_train)
convert_accident_to_bool_val(df_test)

unknown_values = ['–', 'not supported', np.nan]
def set_unknown_fuel_type(df):
    df['fuel_type'] = df['fuel_type'].replace(unknown_values, 'Unknown')

set_unknown_fuel_type(df_train)
set_unknown_fuel_type(df_test)

# Brand processing
t = 800
value_counts = df_train['brand'].value_counts()
other_brands = value_counts[value_counts < t].index

def replace_rare_brands_value(df):
    df['brand'] = df['brand'].apply(lambda x: 'Other' if x in other_brands else x)

replace_rare_brands_value(df_train)
replace_rare_brands_value(df_test)

# Transmission processing
def standardize_transmission(value):
    value = value.lower().strip()  # Normalize case and strip spaces
    if value in ['–', '2', '6 speed at/mt', 'f', '7-speed', '6-speed', 'scheduled for or in production']:
        return 'Unknown'
    if any(v in value for v in ["at", "a/t", "automatic"]):
        match = re.search(r"(\d+)-speed", value)
        return f"Automatic ({match.group(1)}-Speed)" if match else "Automatic"
    if any(v in value for v in ["mt", "m/t", "manual"]):
        match = re.search(r"(\d+)-speed", value)
        return f"Manual ({match.group(1)}-Speed)" if match else "Manual"
    if any(v in value for v in ["cvt", "variable"]):
        return "CVT (Automatic)"
    if "1-speed" in value or "single-speed" in value:
        return "Single-Speed"
    return value

def replace_transmission_vallue(df):
    df['transmission'] = df['transmission'].apply(lambda x: standardize_transmission(x))

replace_transmission_vallue(df_train)
replace_transmission_vallue(df_test)

# Model processing
def split_model(df):
    df[['model_base', 'model_version']] = df['model'].str.split(' ', n=1, expand=True)
    df['model_base'] = df['model_base'].fillna(df['model'])

split_model(df_train)
split_model(df_test)

# Model year processing
def year_to_age(df):
    df['car_age'] = 2024 - df['model_year']

year_to_age(df_train)
year_to_age(df_test)

# Extract Features from engine
def decode_engine(s:str):
    s = s.lower()
    # extract HP
    hp_group = re.match(r'(\d+(\.\d+)?\s*)hp', s )
    engine_hp = float(hp_group.group(1)) if hp_group else None
    # extract cc
    cc_group = re.search(r'(\d+(\.\d+)?\s*)l', s )
    engine_cc = float(cc_group.group(1)) if cc_group else None
    # extract cylinder
    cylinder_group = re.search(r'(\d+(\.\d+)?\s*)cylinder', s )
    engine_cyl = int(cylinder_group.group(1)) if cylinder_group else None

    return engine_hp, engine_cc, engine_cyl

def extract_engine_info(df):
    df[['engine_hp','engine_cc','engine_cyl']]=df['engine'].apply(decode_engine).apply(pd.Series)

extract_engine_info(df_train)
extract_engine_info(df_test)

# Handle missing data of the features extracted from engine and model
imputer = SimpleImputer(strategy='mean')

def fill_missing_num_val(df):
    df['model_version'] = imputer.fit_transform(df[['engine_hp']])
    df['engine_hp'] = imputer.fit_transform(df[['engine_hp']])
    df['engine_cc'] = imputer.fit_transform(df[['engine_cc']])
    df['engine_cyl'] = imputer.fit_transform(df[['engine_cyl']])

fill_missing_num_val(df_train)
fill_missing_num_val(df_test)

# Training
feature_cols = ['brand','model_year','milage','fuel_type','transmission',
                'ext_col','int_col','accident','clean_title','model_base',
                'engine_hp','engine_cc','engine_cyl']

def one_hot_encoding(df, dv, train=True):
    x_dict = df.to_dict(orient='records')
    if train:
        return dv.fit_transform(x_dict)
    else:
        return dv.transform(x_dict)

dv = DictVectorizer(sparse=True)

def split_and_encoding_data(df):
    X = df[feature_cols]
    y = df['price']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = one_hot_encoding(X_train, dv)
    X_val = one_hot_encoding(X_val, dv, False)
    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = split_and_encoding_data(df_train)

model = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=10,
    n_estimators=200,
    nthread=8,
    seed=1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(rmse(y_val, y_pred))

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
