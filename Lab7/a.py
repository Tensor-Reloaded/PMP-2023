import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('auto-mpg.csv')


plt.figure(figsize=(20, 10))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.7)
plt.xlabel('cp')
plt.ylabel('mpg')
plt.savefig('a.png')