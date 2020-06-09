import numpy as np;
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt

# Modifiy the data here for your heatmap.
data = [
    [75, 90, 94, 96, 98], # Output for 50 class classifier
    [67, 84, 89, 94, 96], # Output for 100 class classifier
    [61, 76, 85, 90, 92],
    [55, 74, 82, 86, 90],
    [53, 70, 78, 83, 87],
    [47, 63, 70, 74, 78],
    [39, 53, 60, 64, 69] # Output for 959 class classifier
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
plt.title('Top-N Accuracy for N-Classes (ResNet50)')
plt.xlabel('Top-N Accuracy')
plt.ylabel('Number of Classes')
plt.show() 
