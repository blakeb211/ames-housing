import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
scores = pd.read_csv("scores.csv")
scores.sort_values(by='rmse', ascending=False, inplace=True)

print(scores)
rmse = scores.rmse

fig, ax = plt.subplots()
fig.set_dpi(300)
fig.set_size_inches(w=3, h=2)
x_labels = "Single Tree", "KNN", "Bagged Trees", "Regularized Linear", "Multilayer Perceptron", "Random Forest", "XGBoosted Tree (XG)", "AutoML"

font_param_axes = {'fontname': 'FreeMonoBold', 'fontsize': 12}
font_param_title = {'fontname': 'FreeMonoBold', 'fontsize': 13}
font_param_bars = {'fontname': 'FreeMonoBold', 'fontsize': 10}

# Set up colors
barlist = plt.bar(x_labels, rmse, color='blue')
barlist[6].set_color('orange')
barlist[7].set_color('orange')

# Label axes and title
plt.xticks(ticks=ax.get_xticks(), labels=[])
plt.xlabel('Model', **font_param_axes)
plt.ylabel('RMSE (dollars)', **font_param_axes)
plt.title("Ames Housing Prices Scores",**font_param_title)

# Label bars 
y_text = 3500 
x_text_range = np.linspace(0,6.5,num=8)
for x, lab in zip(x_text_range,x_labels):
    ax.text(x=x,y=y_text,s=lab,rotation=90)

plt.show()


if False:
    sns.set(font_scale=1.5)
    sns.barplot(x=scores.name, y=scores.rmse, hue=hues)
    plt.ylabel("<---- Better RMSE")
    plt.xticks(labels=x_labels, rotation=90, ticks=ax.get_xticks())
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
