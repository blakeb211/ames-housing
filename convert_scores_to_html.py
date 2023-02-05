#!env/bin/python
import pandas as pd

df = pd.read_csv("scores.csv")
df.sort_values(by='rmse',ascending=True,inplace=True)
pd.set_option("display.max.columns",99)
df.to_html("scores.html",index=False)
print("df written to html file")
