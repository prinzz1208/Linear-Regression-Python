
import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LableEncoderr
# define exampleLabel
dataset = pd.read_csv('/home/prinzz/Desktop/Work/Data Science/Assignment/Python/car data.csv')
# print(unique(car_name))
# data=data[0]
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = array(car_name)
# values1 = array(car_name)
# values2 = array(car_name)
# values3 = array(car_name)
# print(values)
# integer encode
car_name_integer = LabelEncoder().fit_transform(dataset["Car_Name"])
fuel_type_integer = LabelEncoder().fit_transform(dataset["Fuel_Type"])
seller_type_integer = LabelEncoder().fit_transform(dataset["Seller_Type"])
trans_integer = LabelEncoder().fit_transform(data["Transmission"])

# type()
data.Car_Name = car_name_integer
data.Fuel_Type = fuel_type_integer
data.Seller_Type = seller_type_integer
data.Transmission = trans_integer

print(data)

# encoder_dict = defaultdict(LabelEncoder)
# labeled_df = data.apply(lambda x: encoder_dict[x.name].fit_transform(x))
# print(labeled_df)
# # binary encode
# # onehot_encoder = LableEncoderr(sparse=False)
# # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# # print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([argmax(integer_encoded[0, :])])
# print(inverted)
