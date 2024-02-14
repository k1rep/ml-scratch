import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(x, 0)
# 画出各函数图形
plt.figure(figsize=(8, 4))
plt.plot(x, sigmoid, c='red', label='sigmoid')
plt.plot(x, tanh, c='red', label='tanh')
plt.plot(x, relu, c='red', label='relu')
plt.xlim(-5, 5)
plt.ylim(-1, 2)
plt.legend(loc='best')
plt.show()

