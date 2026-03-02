import pandas as pd

def data():
    df=pd.read_excel("D:\python\dataSet\converted_data.xlsx")
    x=df.iloc[1:,-1].values
    y=df.iloc[:, -1].values
    print(x)
    return x,y
