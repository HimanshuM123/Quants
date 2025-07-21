#A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array,
# or a table with rows and columns.

import pandas as pd

data ={
    "Categoris" :[100,200,300],
    "Duration":[4,5,6]
}

df = pd.DataFrame(data)
print(df)
print("=================================")
#Locate Row
print(df.loc[0])
print("=================================")
print(df.loc[[0,1]])

#Named Indexes
print("=================================")

n_df = pd.DataFrame(data, index =["day1","day2","day3"])
print(n_df)
print("=================================")
print(n_df.loc["day2"])

print("=================================")
#Read CSV
df_csv = pd.read_csv("bank.csv")
print(df_csv.head())
