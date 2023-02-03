#!env/bin/python
import pandas as pd

df = pd.read_csv("scores.csv")

pd.set_option("display.max.columns",99)
df.to_html("scores.html")
print("df written to html file")
