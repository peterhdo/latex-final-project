import numpy as np;
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt

# Modifiy the data here for your heatmap.
data = [
    [75, 87, 91, 95, 97], # Output for 50 class classifier
    [67, 83, 89, 92, 95], # Output for 100 class classifier
    [58, 75, 83, 88, 92],
    [53, 71, 79, 86, 89],
    [48, 67, 75, 80, 84],
    [44, 58, 66, 71, 74],
    [35, 49, 55, 61, 65] # Output for 959 class classifier
][::-1]
idx = ['50', '100', '150', '200', '250', '500', '959'][::-1]

df = pd.DataFrame(data, columns=[str(i) for i in range(1,6)], index=idx)
p1 = sns.heatmap(df, annot=True)

# fix for matplotlib bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

# Add titles
plt.title('Heatmap of Accuracy Given model size and top-N')
plt.xlabel('Top-N')
plt.ylabel('# of output classes')
plt.show() 