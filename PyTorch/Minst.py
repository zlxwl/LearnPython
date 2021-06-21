import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("mnist_csv\mnist_train.csv")
row = 13
data = df.iloc[row]
label = data[0]
img = np.array(data[1:].values).reshape(28,28)

plt.title("lable = " + str(label))
plt.imshow(img, interpolation="none", cmap="Blues")
plt.show()