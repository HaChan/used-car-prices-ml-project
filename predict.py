import pickle

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# df_res = pd.read_csv("data/sample_submission.csv")
# df_test = pd.read_csv("data/test.csv")

val = {'brand': 'Land', 'car_age': 9, 'milage': 98000, 'fuel_type': 'Gasoline',
 'transmission': 'Automatic (6-Speed)', 'accident': 0, 'clean_title': 1,
 'model_base': 'Rover', 'engine_hp': 240, 'engine_cc': 2.0, 'engine_cyl': 4}
features = dv.transform(val)
y_pred = model.predict(features)
print(y_pred)
