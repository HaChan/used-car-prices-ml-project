# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# # Load data

# %%
df_res = pd.read_csv("data/sample_submission.csv")
df_test = pd.read_csv("data/test.csv")
df_train = pd.read_csv("data/train.csv")

# %% [markdown]
# # EDA

# %% [markdown]
# ## Initial Inspection

# %%
df_res.info()

# %%
df_res.head(5)

# %%
df_res.shape

# %%
df_test.info()

# %%
df_test.head(5)

# %%
df_test.shape

# %%
df_train.info()

# %%
df_train.head(5)

# %%
df_train.shape

# %% [markdown]
# ### Summary Statistics

# %%
df_train.describe()

# %%
df_train.describe(include='object')

# %% [markdown]
# ## Check missing values

# %%
missing_cols = df_train.columns[df_train.isnull().any()]
missing_cols

# %%
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df_train)
plt.show()

# %%
missing_cols = df_test.columns[df_test.isnull().any()]
missing_cols

# %%
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df_test)
plt.show()

# %% [markdown]
# ### Handle missinge values

# %% [markdown]
# #### clean_title

# %%
df_train['clean_title'].unique()


# %%
def convert_clean_title_to_bool_val(df):
    df['clean_title'] = df['clean_title'].map(lambda x: 1 if x == 'Yes' else 0)

convert_clean_title_to_bool_val(df_train)
convert_clean_title_to_bool_val(df_test)

# %%
df_train['clean_title'].unique()

# %%
df_train['clean_title'].value_counts()

# %% [markdown]
# #### accident

# %%
df_train['accident'].unique()


# %%
def convert_accident_to_bool_val(df):
    df['accident'] = df['accident'].map(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

convert_accident_to_bool_val(df_train)
convert_accident_to_bool_val(df_test)

# %%
df_train['accident'].unique()

# %%
df_train['accident'].value_counts()

# %% [markdown]
# #### fuel_type

# %%
df_train['fuel_type'].unique()

# %%
unknown_values = ['–', 'not supported', np.nan]
def set_unknown_fuel_type(df):
    df['fuel_type'] = df['fuel_type'].replace(unknown_values, 'Unknown')

set_unknown_fuel_type(df_train)
set_unknown_fuel_type(df_test)

# %%
df_train['fuel_type'].unique()

# %%
df_train['fuel_type'].value_counts()

# %% [markdown]
# ## Univariate Analysis

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Price distribution
sns.histplot(df_train['price'], kde=True)
plt.title('Price Distribution')
plt.show()

# %%
ax = sns.kdeplot(df_train['price'], color='blue', fill=True, log_scale=True)
ax.set(xlabel='log price');

# %%
# Mileage distribution
sns.boxplot(x=df_train['milage'])
plt.title('Mileage Distribution')
plt.show()

# %%
sns.countplot(x='fuel_type', data=df_train)
plt.title('Fuel Type Distribution')
plt.show()

# %%
sns.countplot(x='transmission', data=df_train)
plt.title('Transmission Type Distribution')
plt.xticks(rotation=90)
plt.show()

# %%
sns.countplot(x='brand', data=df_train)
plt.title('Brand Distribution')
plt.xticks(rotation=90)
plt.show()

# %%
df_train['ext_col'].unique()

# %%
df_train['ext_col'].value_counts()

# %%
df_train['int_col'].unique()

# %% [markdown]
# ## Bivariate Analysis

# %%
sns.scatterplot(data=df_train, x='milage', y='price', color='orange')

# %%
sns.scatterplot(data=df_train, x='model_year', y='price', color='blue')

# %%
sns.boxplot(x='fuel_type', y='price', data=df_train)
plt.title('Fuel Type vs Price')
plt.show()

# %%
sns.boxplot(x='transmission', y='price', data=df_train)
plt.title('Transmission vs Price')
plt.xticks(rotation=90)
plt.show()

# %%
sns.boxplot(x='brand', y='price', data=df_train)
plt.title('Brand vs Price')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### correlations among numerical variables

# %%
corr = df_train[['price', 'milage', 'model_year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# %%
sns.boxplot(df_train['price'])
plt.title('Price Outliers')
plt.show()

from scipy.stats import zscore
df_train['price_zscore'] = zscore(df_train['price'])
outliers = df_train[df_train['price_zscore'] > 3]
print(len(outliers))

# %%
sns.boxplot(df_train['milage'])
plt.title('Milage Outliers')
plt.show()

from scipy.stats import zscore
df_train['milage_zscore'] = zscore(df_train['milage'])
outliers = df_train[df_train['milage_zscore'] > 3]
print(len(outliers))

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Brand processing
#
# Simplify brand value that has value counts < some threshold will be set to other

# %%
df_train['brand'].value_counts()

# %%
brand_counts = df_train['brand'].value_counts()
rare_brands = brand_counts[brand_counts < 1000].index
rare_brands

# %%
t = 800
value_counts = df_train['brand'].value_counts()
other_brands = value_counts[value_counts < t].index

def replace_rare_brands_value(df):
    df['brand'] = df['brand'].apply(lambda x: 'Other' if x in other_brands else x)


# Apply the function to both DataFrames
replace_rare_brands_value(df_train)
replace_rare_brands_value(df_test)

# Check the distribution of the updated 'brand' column
print("Updated df brand distribution:")
print(df_train['brand'].value_counts())


# %% [markdown]
# ### Transmission processing
#
# Standardize the transmission value

# %%
df_train['transmission'].value_counts()

# %%
df_train['transmission'].unique()

# %%
import re

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

# %%
df_train['transmission'].unique()

# %% [markdown]
# ### Model processing
#
# Split model to base and version

# %%
df_train['model'].value_counts()

# %%
df_train['model'].str.split(' ', n=1, expand=True)


# %%
def split_model(df):
    df[['model_base', 'model_version']] = df['model'].str.split(' ', n=1, expand=True)
    df['model_base'] = df['model_base'].fillna(df['model'])

split_model(df_train)
split_model(df_test)

# %%
print(df_train['model_base'].value_counts())
print(df_train['model_version'].value_counts())


# %% [markdown]
# ### Convert model_year into car_age

# %%
def year_to_age(df):
    df['car_age'] = 2024 - df['model_year']

year_to_age(df_train)
year_to_age(df_test)

# %%
df_train['car_age'].unique()

# %% [markdown]
# ### Extract Features from engine

# %%
df_train['engine'].value_counts()

# %%
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

# %%
df_train['engine_hp'].value_counts()

# %%
df_engine = df_train[['engine_hp','engine_cc','engine_cyl','fuel_type']]

# %%
missing_cols = df_engine.columns[df_engine.isnull().any()]
missing_cols

# %%
df_test_engine = df_test[['engine_hp','engine_cc','engine_cyl']]
missing_cols = df_test_engine.columns[df_test_engine.isnull().any()]
missing_cols

# %%
missing_cols = df_train.columns[df_train.isnull().any()]
missing_cols

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

def fill_missing_num_val(df):
    df['model_version'] = imputer.fit_transform(df[['engine_hp']])
    df['engine_hp'] = imputer.fit_transform(df[['engine_hp']])
    df['engine_cc'] = imputer.fit_transform(df[['engine_cc']])
    df['engine_cyl'] = imputer.fit_transform(df[['engine_cyl']])

fill_missing_num_val(df_train)
fill_missing_num_val(df_test)

# %%
missing_cols = df_train.columns[df_train.isnull().any()]
missing_cols

# %% [markdown]
# # Model training

# %%
feature_cols = ['brand','car_age','milage','fuel_type','transmission',
                'accident','clean_title','model_base',
                'engine_hp','engine_cc','engine_cyl']

data_train = df_train[feature_cols]
y_train = df_train['price']

# %%
data_train.info()

# %%
dv = DictVectorizer(sparse=True)


# %%
def one_hot_encoding(df, dv, train=True):
    x_dict = df.to_dict(orient='records')
    if train:
        return dv.fit_transform(x_dict)
    else:
        return dv.transform(x_dict)


# %%
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

# %%
X_train = one_hot_encoding(data_train, dv)

# %%
rf.fit(X_train, y_train)

# %%
X_test = df_test[feature_cols]
X_test = one_hot_encoding(df_test, dv, False)
y_test = df_res['price']

# %%
test_pred = rf.predict(X_test)

# %%
test_pred

# %%
print(rmse(y_test, test_pred))
print(rmse(test_pred, df_res['price']))

# %% [markdown]
# # XGB

# %%
import xgboost as xgb

# %%
features = list(dv.get_feature_names_out())

# %%
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

# %%
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# %%
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# %%
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

# %%
model = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=5, evals=watchlist)

# %%
y_pred = model.predict(dtest)

# %%
rmse(y_test, y_pred)

# %%


# %% [markdown]
# # Train model script
#
#

# %%
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# %%
df_res = pd.read_csv("data/sample_submission.csv")
df_test = pd.read_csv("data/test.csv")
df_train = pd.read_csv("data/train.csv")

# %%
t = 800
value_counts = df_train['brand'].value_counts()
other_brands = value_counts[value_counts < t].index
def replace_rare_brands_value(df):
    df['brand'] = df['brand'].apply(lambda x: 'Other' if x in other_brands else x)

replace_rare_brands_value(df_train)
replace_rare_brands_value(df_test)

def standardize_transmission(value):
    value = value.lower().strip()  # Normalize case and strip spaces
    if value in ['–', '2', '6 speed at/mt', 'f', '7-speed', '6-speed', 'scheduled for or in production']:
        return 'Unknown'
    # Automatic transmissions
    if any(v in value for v in ["at", "a/t", "automatic"]):
        match = re.search(r"(\d+)-speed", value)
        return f"Automatic ({match.group(1)}-Speed)" if match else "Automatic"
    # Manual transmissions
    if any(v in value for v in ["mt", "m/t", "manual"]):
        match = re.search(r"(\d+)-speed", value)
        return f"Manual ({match.group(1)}-Speed)" if match else "Manual"
    # CVT transmissions
    if any(v in value for v in ["cvt", "variable"]):
        return "CVT (Automatic)"
    # Single-Speed transmissions
    if "1-speed" in value or "single-speed" in value:
        return "Single-Speed"
    return value

def replace_transmission_vallue(df):
    df['transmission'] = df['transmission'].apply(lambda x: standardize_transmission(x))

replace_transmission_vallue(df_train)
replace_transmission_vallue(df_test)

unknown_values = ['–', 'not supported', np.nan]
def set_unknown_fuel_type(df):
    df['fuel_type'] = df['fuel_type'].replace(unknown_values, 'Unknown')

set_unknown_fuel_type(df_train)
set_unknown_fuel_type(df_test)

def split_model(df):
    df[['model_base', 'model_version']] = df['model'].str.split(' ', n=1, expand=True)
    df['model_base'] = df['model_base'].fillna(df['model'])

split_model(df_train)
split_model(df_test)

def decode_engine(s:str):
    s = s.lower()
    # extract HP
    hp_group = re.match(r'(\d+(\.\d+)?\s*)hp', s )
    engine_hp = float(hp_group.group(1)) if hp_group else None
    # extract cc
    cc_group = re.search(r'(\d+(\.\d+)?\s*)l', s )
    engine_cc = float(cc_group.group(1)) if cc_group else None
    # extract cylinder cnt
    cylinder_group = re.search(r'(\d+(\.\d+)?\s*)cylinder', s )
    engine_cyl = int(cylinder_group.group(1)) if cylinder_group else None

    return engine_hp, engine_cc, engine_cyl

def extract_engine_info(df):
    df[['engine_hp','engine_cc','engine_cyl']]=df['engine'].apply(decode_engine).apply(pd.Series)

extract_engine_info(df_train)
extract_engine_info(df_test)

def convert_clean_title_to_bool_val(df):
    df['clean_title'] = df['clean_title'].map(lambda x: 1 if x == 'Yes' else 0)

convert_clean_title_to_bool_val(df_train)
convert_clean_title_to_bool_val(df_test)

def convert_accident_to_bool_val(df):
    df['accident'] = df['accident'].map(lambda x: 1 if x == 'Yes' else 0)

convert_accident_to_bool_val(df_train)
convert_accident_to_bool_val(df_test)

imputer = SimpleImputer(strategy='mean')

def fill_missing_num_val(df):
    df['engine_hp'] = imputer.fit_transform(df[['engine_hp']])
    df['engine_cc'] = imputer.fit_transform(df[['engine_cc']])
    df['engine_cyl'] = imputer.fit_transform(df[['engine_cyl']])

fill_missing_num_val(df_train)
fill_missing_num_val(df_test)

# %%
feature_cols = ['brand','model_year','milage','fuel_type','transmission',
                'ext_col','int_col','accident','clean_title','model_base',
                'engine_hp','engine_cc','engine_cyl']

# %%
def one_hot_encoding(df, dv, train=True):
    x_dict = df.to_dict(orient='records')
    if train:
        return dv.fit_transform(x_dict)
    else:
        return dv.transform(x_dict)


# %%
dv = DictVectorizer(sparse=True)

def split_and_encoding_data(df):
    X = df[feature_cols]
    y = df['price']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = one_hot_encoding(X_train, dv)
    X_val = one_hot_encoding(X_val, dv, False)
    return X_train, X_val, y_train, y_val

# %%
X_train, X_val, y_train, y_val = split_and_encoding_data(df_train)


# %%
def rand_forest(X_train, y_train, n_est, rand_state):
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=10, random_state=rand_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def rf_rmse_score(rf, X_val, y_val):
    val_pred = rf.predict(X_val)
    return rmse(y_val, val_pred)


# %%
rf10 = rand_forest(X_train, y_train, 10, 42)

# %%
print(rf_rmse_score(rf10, X_val, y_val))

# %%
rf50 = rand_forest(X_train, y_train, 50, 42)

# %%
print(rf_rmse_score(rf50, X_val, y_val))

# %%
rf100 = rand_forest(X_train, y_train, 100, 42)

# %%
print(rf_rmse_score(rf100, X_val, y_val))

# %%
import xgboost as xgb

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_val, label=y_val, feature_names=features)
watchlist = [(dtrain, 'train'), (dtest, 'test')]

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100, verbose_eval=5, evals=watchlist)
y_pred = model.predict(dtest)

rmse(y_val, y_pred)


# %%
