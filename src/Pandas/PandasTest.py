import pandas as pd
a =[1,2,3]
myVar = pd.Series(a)
print(myVar)

#Create Labels
myLabels = pd.Series(a, index=["x","y","z"])
print(myLabels)