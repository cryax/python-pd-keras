# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
t=np.linspace(-10,10,1)
def x(i):
    if i <= 0:
        j = -1
    else :
        j = 1
    return j
y = []

for i in t:
    y.append(x(i))

# or using list comprehensions
#y = [8*x(i)-4*x(i/2)-3*x(i*8) for i in t]
axes = plt.gca()
axes.set_xlim([-10,10])
axes.set_ylim([-2,2])
plt.plot(t,y)
plt.ylabel('y')
plt.xlabel('t')
plt.show()