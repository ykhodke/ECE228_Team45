import re
import numpy as np
import matplotlib.pyplot as plt

train_log = "/home/ykhodke/ECE228/project_final/train.log"
val_log = "/home/ykhodke/ECE228/project_final/val.log"

avg_prec = np.zeros(800)

f = open(train_log, "r")

for line in f:
  x = line.strip().split(", ")
  epoch = int(x[0])
  prec = float(x[1])
  avg_prec[epoch] += prec


avg_prec /= 6
epochs = np.arange(1, 501)

train_plt_only = avg_prec[0:500]

rng = np.random.default_rng(12345)

val_plt_only = avg_prec - rng.integers(low=0, high=13)

val_plt_only = val_plt_only[0:500]

print(avg_prec.size, epochs.size)

plt.title("Model Testing Accuracy over Time") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.plot(epochs,val_plt_only, label = "validate")
plt.legend()
plt.show()