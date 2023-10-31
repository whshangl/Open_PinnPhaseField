import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Fig_path = Path('./Figures')
Fig_path.mkdir(parents=True, exist_ok=True)

############################################################################################################
### data source: https://www.nature.com/articles/nature16463 (Source data to Fig. 2)
### center (472, 187), x range (404, 540), y range (119, 255), unit in pixel, using data filter in Excel: 404 <= x <= 540, 119 <= y <= 255
############################################################################################################


############ plot raw data ############

# exp_data = np.loadtxt("./experimental_data/experimental_data_part.txt",  comments='%', delimiter=None, skiprows=0)

# x = exp_data[:,0:1]
# y = exp_data[:,1:2]

# Px = exp_data[:,2:3]
# Py = exp_data[:,3:4]



# print(np.max(np.abs(Px)), np.max(np.abs(Py)))

# plt.gca().set_aspect('equal', adjustable='box')
# plt.quiver(x, y, Px, Py)
# plt.xlim((404,540))
# plt.ylim((119,255))
# plt.show()
# plt.savefig("./Figures/fig_experimental_domain.pdf")


############ transfer unit in pixel to unit in nm ############
### relation: center (472, 187) in pixel ==> center (0, 0) in nm
### relation: length 136 (= 540- 404 = 255 - 119) pixels ==> length 4 nm ==> 34 pixels = 1 nm
# X, Y = (x-472)/34, (y-187)/34
# exp_single_data = np.zeros((x.shape[0],4))
# exp_single_data[:, 0:1] = X
# exp_single_data[:, 1:2] = Y 
# exp_single_data[:, 2:3] = Px
# exp_single_data[:, 3:4] = Py
# np.save("./experimental_data/experimental_data_domain.npy", exp_single_data)

exp_data = np.load('./experimental_data/experimental_data_domain.npy')


X = exp_data[:,0:1]
Y = exp_data[:,1:2]

Px = exp_data[:,2:3]
Py = exp_data[:,3:4]


#### print X, Y  position
# for x, y in zip(X, Y):
#     print(x[0],y[0])


plt.rcParams['axes.facecolor'] = 'black'
fig = plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal', adjustable='box')
plt.quiver(X, Y, Px, Py, color = 'yellow')
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.xticks(np.arange(-2,2.1,1))
plt.yticks(np.arange(-2,2.1,1))
plt.xlabel("$x_1$ [nm]", fontsize = 16)
plt.ylabel("$x_2$ [nm]", fontsize = 16)
plt.title("Polar displacement", fontsize = 18)
# plt.show()
plt.savefig('./Figures/experimental_data_plot_colorful.pdf')








   