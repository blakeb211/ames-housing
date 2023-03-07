import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager

# Print font options
if False:
    font_names = sorted(font_manager.get_font_names())
    print(f"font names {font_names}")


scores = pd.read_csv("scores.csv")
scores.sort_values(by='rmse', ascending=False, inplace=True)

rmse = scores.rmse / 1000

fig, ax = plt.subplots()
fig.set_dpi(300)
fig.set_size_inches(w=5, h=3)
size_w, size_h = fig.get_size_inches()
print(f"size of pic after setting: {size_w} x {size_h}")
x_labels = "Single Tree", "KNN", "Bagged Trees", "Regularized Linear", "Neural Net", "Random Forest", "XGBoosted Tree", "AutoML"

# Only worry about fiddling with these 3
font_param_axes = {'fontname': 'Liberation Serif', 'fontsize': 11}
font_param_title = {'fontname': 'Liberation Serif', 'fontsize': 12}
font_param_barlabels = {'fontname': 'Liberation Serif', 'fontsize': 10}

# No need to fiddle
font_param_yticks = font_param_axes 
font_param_yticks['fontsize'] = font_param_axes['fontsize']-2

# Bar colors 
barlist = plt.bar(x_labels, rmse, color='blue')
barlist[6].set_color('orange')
barlist[7].set_color('orange')

# Label axes and title
plt.xticks(ticks=ax.get_xticks(), labels=[])
plt.xlabel('Model', **font_param_axes)
plt.ylabel('RMSE (thousands of dollars)', **font_param_axes)
plt.yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels())
plt.title("Housing Price Prediction Error",**font_param_title)

def add_labels_1():
    # Label bars 
    text_x_offset=0.12
    y_text = 3 
    x_text_range = range(len(x_labels))

    for x, lab in zip(ax.get_xticks(),x_labels):
        ax.text(x=x-text_x_offset,y=y_text,s=lab,rotation=90,color='white', **font_param_barlabels)

add_labels_1()
ax.legend()
plt.show()


if False:
    sns.set(font_scale=1.5)
    sns.barplot(x=scores.name, y=scores.rmse, hue=hues)
    plt.ylabel("<---- Better RMSE")
    plt.xticks(labels=x_labels, rotation=90, ticks=ax.get_xticks())
    plt.xlabel("Estimator")
    plt.show()

