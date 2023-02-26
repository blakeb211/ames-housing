import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
scores = pd.read_csv("scores.csv")
scores.sort_values(by='rmse',ascending=False,inplace=True)

print(scores)
rmse = scores.rmse
sns.set(font_scale=1.5)

fig, ax = plt.subplots()
hues = list(6*["Sklearn"])
hues += ["Companion Lib"]
hues += ["Companion Lib"]

x_labels = "Single Tree", "KNN", "Bagged Trees","Regularized Linear", "Neural Net", "Forest", "Boosted Tree", "AutoML"
print(len(scores.name))
sns.barplot(x=scores.name,y=scores.rmse,hue=hues)
plt.ylabel("<---- Better RMSE")
plt.xticks(labels=x_labels,rotation=90, ticks=ax.get_xticks())
plt.xlabel("Estimator")
plt.show()

if False:
    # import the necessary python packages
    import pandas as pd
    import seaborn as sns
    import numpy as np
    
    # read the dataset using pandas read_csv
    # function
    data = pd.read_csv(r"path to\tips.csv")
    
    # group the multilevel categorical
    # values and flatten the index
    groupedvalues = data.groupby('day').sum().reset_index()
    
    # define the color palette of different colors
    pal = sns.color_palette("Greens_d", len(groupedvalues))
    # use argsort method
    rank = groupedvalues["total_bill"].argsort()
    
    # use dataframe grouped by days to plot a
    # bar chart between days and total bill
    ax = sns.barplot(x='day', y='total_bill',
                    data=groupedvalues,
                    palette=np.array(pal[::-1])[rank])
    
    # now use a for loop to iterate through
    # each row of the grouped dataframe
    # assign bar value  to each row
    for index, row in groupedvalues.iterrows():
        ax.text(row.name, row.tip, round(row.total_bill, 2),
                color='white', ha='center')
