import pandas as pd
filename='E:\csvfile1.csv'
csv_read=pd.read_csv(filename)
print(filename)
#printimg first five lines of the file
print(csv_read.head())
#printing the last 5 lines
print(csv_read.tail())
#displaying the info 
print(csv_read.info())
print(csv_read.shape)
print(csv_read.describe())
print(csv_read.columns)
print(csv_read.index)
print(csv_read.dtypes)
