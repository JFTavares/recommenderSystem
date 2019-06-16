import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/ml-1m/ratings.dat')
data.head()
data.rating.value_counts()
data.rating.value_counts().plot(kind='bar')
plt.show()