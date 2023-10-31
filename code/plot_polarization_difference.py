import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

Fig_path = Path('./Figures')
Fig_path.mkdir(parents=True, exist_ok=True)

#### FEM data
FEM_data_pinn_coef = np.loadtxt('./FEM_data/P1P2_FEM_from_pinn_coef.txt',  comments='%', delimiter=None, skiprows=0)
FEM_data_paper_coef = np.loadtxt('./FEM_data/P1P2_FEM_from_paper_coef.txt',  comments='%', delimiter=None, skiprows=0)

#### experimental data: polar displacement
exp_data = np.load('./experimental_data/experimental_data_domain.npy')

#### experimental data: polarization, assume the coefficient to link the polar displacement and polarization is 0.91
Px_exp = exp_data[:, 2:3] * 0.91
Py_exp = exp_data[:, 3:4] * 0.91


##### plot difference
X = exp_data[:,0:1]
Y = exp_data[:,1:2]

dif_Px_paper_exp = FEM_data_paper_coef[:, 2:3] - Px_exp
dif_Py_paper_exp = FEM_data_paper_coef[:, 3:4] - Py_exp

dif_Px_pinn_exp = FEM_data_pinn_coef[:, 2:3] - Px_exp
dif_Py_pinn_exp = FEM_data_pinn_coef[:, 3:4] - Py_exp


# ### paper coefficients
# # fig = plt.figure(figsize=(6, 6))
# # plt.gca().set_aspect('equal', adjustable='box')
# # plt.quiver(X, Y, dif_Px_paper_exp, dif_Py_paper_exp)
# # plt.xlim((-2,2))
# # plt.ylim((-2,2))
# # plt.xlabel("$x_1$ [nm]", fontsize = 16)
# # plt.ylabel("$x_2$ [nm]", fontsize = 16)
# # plt.show()

# ### pinn coefficients
# # fig = plt.figure(figsize=(6, 6))
# # plt.gca().set_aspect('equal', adjustable='box')
# # plt.quiver(X, Y, dif_Px_pinn_exp, dif_Py_pinn_exp)
# # plt.xlim((-2,2))
# # plt.ylim((-2,2))
# # plt.xlabel("$x_1$ [nm]", fontsize = 16)
# # plt.ylabel("$x_2$ [nm]", fontsize = 16)
# # plt.show()

##### three figs in one plot
fig, (fig_pinn, fig_exp, fig_paper) = plt.subplots(1, 3, figsize = (20, 6))

fig_pinn.quiver(X, Y, dif_Px_pinn_exp, dif_Py_pinn_exp, scale = 2, scale_units = 'inches', color = 'red')
fig_pinn.set_aspect('equal', adjustable='box')
fig_pinn.set_xlim((-2,2))
fig_pinn.set_ylim((-2,2))
fig_pinn.set_xticks(np.arange(-2,2.1,1))
fig_pinn.set_yticks(np.arange(-2,2.1,1))
fig_pinn.set_xlabel("$x_1$ [nm]", fontsize = 14)
fig_pinn.set_ylabel("$x_2$ [nm]", fontsize = 14)
fig_pinn.set_title("Difference: coefficients from a PINN", fontsize = 17)


fig_exp.quiver(X, Y, Px_exp, Py_exp, scale = 2, scale_units = 'inches', color = 'yellow')
fig_exp.set_aspect('equal', adjustable='box')
fig_exp.set_xlim((-2,2))
fig_exp.set_ylim((-2,2))
fig_exp.set_xticks(np.arange(-2,2.1,1))
fig_exp.set_yticks(np.arange(-2,2.1,1))
fig_exp.set_xlabel("$x_1$ [nm]", fontsize = 14)
fig_exp.set_ylabel("$x_2$ [nm]", fontsize = 14)
fig_exp.set_title("Polarization: experimental dataset", fontsize = 17)
fig_exp.set_facecolor("black")

fig_paper.quiver(X, Y, dif_Px_paper_exp, dif_Py_paper_exp, scale = 2, scale_units = 'inches', color = 'red')
fig_paper.set_aspect('equal', adjustable='box')
fig_paper.set_xlim((-2,2))
fig_paper.set_ylim((-2,2))
fig_paper.set_xticks(np.arange(-2,2.1,1))
fig_paper.set_yticks(np.arange(-2,2.1,1))
fig_paper.set_xlabel("$x_1$ [nm]", fontsize = 14)
fig_paper.set_ylabel("$x_2$ [nm]", fontsize = 14)
fig_paper.set_title("Difference: coefficients from literature", fontsize = 17)
# plt.show()
plt.savefig('./Figures/polarization_difference_plot_colorful.pdf', bbox_inches = 'tight')

