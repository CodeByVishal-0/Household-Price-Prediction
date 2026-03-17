from preprocessing import preprocessings
import pandas as pd

housing=pd.read_csv('files/housing.csv')
arr=preprocessings.fit_transform(housing)
clm=preprocessings.get_feature_names_out()

print(arr)
print(clm)