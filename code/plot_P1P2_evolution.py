import numpy as np
from pathlib import Path

x1, y1 = 1.0, 0.0
x2, y2 = 0.0, 1.0
x3, y3 = -1.0, 0.0
x4, y4 = 0.0, -1.0

Fig_path = Path('./Figures')
Fig_path.mkdir(parents=True, exist_ok=True)

##### combine data of the point (x1, y1) #####
pinn_P1P2_x1y1_0to5   = np.load("./PINN_data/pinn_P1P2_0to5_" + str(x1) + "_" + str(y1) + ".npy")
pinn_P1P2_x1y1_5to15  = np.load("./PINN_data/pinn_P1P2_5to15_" + str(x1) + "_" + str(y1) + ".npy")
pinn_P1P2_x1y1_15to25 = np.load("./PINN_data/pinn_P1P2_15to25_" + str(x1) + "_" + str(y1) + ".npy")
pinn_P1P2_x1y1 = np.vstack((pinn_P1P2_x1y1_0to5, pinn_P1P2_x1y1_5to15, pinn_P1P2_x1y1_15to25))

##### combine data of the point (x2, y2) #####
pinn_P1P2_x2y2_0to5   = np.load("./PINN_data/pinn_P1P2_0to5_" + str(x2) + "_" + str(y2) + ".npy")
pinn_P1P2_x2y2_5to15  = np.load("./PINN_data/pinn_P1P2_5to15_" + str(x2) + "_" + str(y2) + ".npy")
pinn_P1P2_x2y2_15to25 = np.load("./PINN_data/pinn_P1P2_15to25_" + str(x2) + "_" + str(y2) + ".npy")
pinn_P1P2_x2y2 = np.vstack((pinn_P1P2_x2y2_0to5, pinn_P1P2_x2y2_5to15, pinn_P1P2_x2y2_15to25))

##### combine data of the point (x3, y3) #####
pinn_P1P2_x3y3_0to5   = np.load("./PINN_data/pinn_P1P2_0to5_" + str(x3) + "_" + str(y3) + ".npy")
pinn_P1P2_x3y3_5to15  = np.load("./PINN_data/pinn_P1P2_5to15_" + str(x3) + "_" + str(y3) + ".npy")
pinn_P1P2_x3y3_15to25 = np.load("./PINN_data/pinn_P1P2_15to25_" + str(x3) + "_" + str(y3) + ".npy")
pinn_P1P2_x3y3 = np.vstack((pinn_P1P2_x3y3_0to5, pinn_P1P2_x3y3_5to15, pinn_P1P2_x3y3_15to25))

##### combine data of the point (x4, y4) #####
pinn_P1P2_x4y4_0to5   = np.load("./PINN_data/pinn_P1P2_0to5_" + str(x4) + "_" + str(y4) + ".npy")
pinn_P1P2_x4y4_5to15  = np.load("./PINN_data/pinn_P1P2_5to15_" + str(x4) + "_" + str(y4) + ".npy")
pinn_P1P2_x4y4_15to25 = np.load("./PINN_data/pinn_P1P2_15to25_" + str(x4) + "_" + str(y4) + ".npy")
pinn_P1P2_x4y4 = np.vstack((pinn_P1P2_x4y4_0to5, pinn_P1P2_x4y4_5to15, pinn_P1P2_x4y4_15to25))


##### FEM data
fem_P1_x1y1 = np.loadtxt('./FEM_data/P1_x1y1_251steps.txt', comments='%',delimiter=None)
fem_P1_x2y2 = np.loadtxt('./FEM_data/P1_x2y2_251steps.txt', comments='%',delimiter=None)
fem_P1_x3y3 = np.loadtxt('./FEM_data/P1_x3y3_251steps.txt', comments='%',delimiter=None)
fem_P1_x4y4 = np.loadtxt('./FEM_data/P1_x4y4_251steps.txt', comments='%',delimiter=None)

fem_P2_x1y1 = np.loadtxt('./FEM_data/P2_x1y1_251steps.txt', comments='%',delimiter=None)
fem_P2_x2y2 = np.loadtxt('./FEM_data/P2_x2y2_251steps.txt', comments='%',delimiter=None)
fem_P2_x3y3 = np.loadtxt('./FEM_data/P2_x3y3_251steps.txt', comments='%',delimiter=None)
fem_P2_x4y4 = np.loadtxt('./FEM_data/P2_x4y4_251steps.txt', comments='%',delimiter=None)


import matplotlib.pyplot as plt 


fig = plt.figure(figsize=(8, 7))

### PINN, P1
# plt.plot(pinn_P1P2_x1y1[:,2], pinn_P1P2_x1y1[:,3],  color = 'seagreen',  linestyle = "dotted", linewidth = 2.0)
# plt.plot(pinn_P1P2_x2y2[:,2], pinn_P1P2_x2y2[:,3],  color = 'seagreen',  linestyle = "dotted", linewidth = 2.0)
# plt.plot(pinn_P1P2_x3y3[:,2], pinn_P1P2_x3y3[:,3],  color = 'seagreen',    linestyle = "dotted", linewidth = 2.0)
# plt.plot(pinn_P1P2_x4y4[:,2], pinn_P1P2_x4y4[:,3],  color = 'seagreen',      linestyle = "dotted", linewidth = 2.0)

### FEM, P1
# plt.plot(fem_P1_x1y1[:,0], fem_P1_x1y1[:,1],  color = 'sandybrown', linestyle = "dashed", linewidth = 2.0)
# plt.plot(fem_P1_x2y2[:,0], fem_P1_x2y2[:,1],  color = 'sandybrown', linestyle = "dashed", linewidth = 2.0)
# plt.plot(fem_P1_x3y3[:,0], fem_P1_x3y3[:,1],  color = 'sandybrown',   linestyle = "dashed", linewidth = 2.0)
# plt.plot(fem_P1_x4y4[:,0], fem_P1_x4y4[:,1],  color = 'sandybrown',     linestyle = "dashed", linewidth = 2.0)


#### legend, P1
# plt.text(4.0, -0.07, 'A (1,0)')
# plt.text(4.0, 0.16, 'B (0,1)')
# plt.text(4.0, 0.03, 'C (-1,0)')
# plt.text(4.0, -0.2, 'D (0,-1)')


#### P1
# plt.ylabel('$P_1$ [(aC)(nm)$^{-2}$]', fontsize = 14)
# plt.xlabel('t [(aC)$^2$(aJ)$^{-1}$(nm)$^{-1}$]', fontsize = 14)


### PINN, P2
plt.plot(pinn_P1P2_x1y1[:,2], pinn_P1P2_x1y1[:,4],  color = 'seagreen',  linestyle = "dotted", linewidth = 2.5)
plt.plot(pinn_P1P2_x2y2[:,2], pinn_P1P2_x2y2[:,4],  color = 'seagreen',  linestyle = "dotted", linewidth = 2.5)
plt.plot(pinn_P1P2_x3y3[:,2], pinn_P1P2_x3y3[:,4],  color = 'seagreen',    linestyle = "dotted", linewidth = 2.5)
plt.plot(pinn_P1P2_x4y4[:,2], pinn_P1P2_x4y4[:,4],  color = 'seagreen',      linestyle = "dotted", linewidth = 2.5)

### FEM, P2
plt.plot(fem_P2_x1y1[:,0], fem_P2_x1y1[:,1],  color = 'sandybrown', linestyle = "dashed", linewidth = 2.0)
plt.plot(fem_P2_x2y2[:,0], fem_P2_x2y2[:,1],  color = 'sandybrown', linestyle = "dashed", linewidth = 2.0)
plt.plot(fem_P2_x3y3[:,0], fem_P2_x3y3[:,1],  color = 'sandybrown',   linestyle = "dashed", linewidth = 2.0)
plt.plot(fem_P2_x4y4[:,0], fem_P2_x4y4[:,1],  color = 'sandybrown',     linestyle = "dashed", linewidth = 2.0)


#### legend, P2
plt.text(4.0, -0.07, 'D (0,-1)')
plt.text(4.0, 0.16,  'C (-1,0)')
plt.text(4.0, 0.03,  'B (0,1)')
plt.text(4.0, -0.2,  'A (1,0)')

#### P2
plt.ylabel('$P_2$ [(aC)(nm)$^{-2}$]', fontsize = 14)
plt.xlabel('t [(aC)$^2$(aJ)$^{-1}$(nm)$^{-1}$]', fontsize = 14)

#### shared by P1 and P2
plt.plot([10, 11.3], [0.7,  0.7], linestyle = "dotted", color = 'seagreen', linewidth = 2.0)
plt.plot([4, 5.3], [0.7,  0.7], linestyle = "dashed", color = 'sandybrown', linewidth = 2.0)
plt.text(11.5, 0.68, 'PINN')
plt.text(5.5, 0.68, 'FEM')


plt.savefig('./Figures/P2_evolution.pdf')


