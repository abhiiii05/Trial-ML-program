import pandas as pd


filename='Housedata.csv'
filepath=pd.read_csv(filename)

filepath.columns

Data=filepath.dropna(axis=0)

dependent_var=Data.price

Data_Features=['bedrooms','bathrooms','floors','lat','long']

independent_var=Data[Data_Features]


from sklearn.tree import DecisionTreeClassifier 
model=DecisionTreeClassifier(random_state=1)

model.fit(independent_var,dependent_var)





print("Predecting the price of random 5 houses")
print("These are the houses")
print(independent_var.head())
print("Now , predicting the price of the houses:")
print(model.predict(independent_var.head()))


