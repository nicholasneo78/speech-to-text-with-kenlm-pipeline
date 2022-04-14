import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.Series([1,3,5,5,5,3,7,6])
# print(data)

data.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Distribution')
plt.xlabel('Samples')
plt.ylabel('Number of inputs')
plt.grid(axis='y', alpha=0.75)

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()