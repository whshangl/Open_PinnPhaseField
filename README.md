# Open_PinnPhaseField
The code depends on the deep learning package [Pytorch](https://pytorch.org/) and [DeepXDE](https://deepxde.readthedocs.io/en/latest/). [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is the recommended package manager since it has all dependencies. Once equipped with [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), one can install [Pytorch](https://pytorch.org/) and [DeepXDE](https://deepxde.readthedocs.io/en/latest/) using

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

`conda install -c conda-forge deepxde`

***To finish the training in several hours, a computer with GPU acceleration is necessary***. 
>[FEM_data](https://github.com/whshangl/Open_PinnPhaseField/tree/main/code/FEM_data): a folder collecting files of FEM data.

>[experimental_data](https://github.com/whshangl/Open_PinnPhaseField/tree/main/code/experimental_data): a folder collecting files of experimental data. 

> [pinn_time5.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/code/pinn_time5.py): PINN trained for the phase-field model of ferroelectric microstructure evolution when the time interval is [0, 5].

> [pinn_time10.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_time10.py): PINN trained for the phase-field model of ferroelectric microstructure evolution when the time interval is [0, 10].

> [pinn_5plus10.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_5plus10.py): PINN trained for the time interval [5, 15]. Run this one after runing [pinn_time5.py](https://github.com/whshangl/PINN_PhaseField/blob/main/Code/pinn_time5.py).

> [pinn_5plus10plus10.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_5plus10plus10.py): PINN trained for the time interval [15, 25]. Run this one after running [pinn_time5.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_time5.py) and [pinn_5plus10.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_5plus10.py).

> [FEM_PINN_comparison.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/FEM_PINN_comparison.py): plot figures that compare FEM and PINN results.

> [plot_P1P2_evolution.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/plot_P1P2_evolution.py): plot figures that show the polarization evolution over time.

> [pinn_inverse_2param.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_inverse_2param.py): PINN trained for the inverse problem to infer two of the gradient energy coefficients using labeled data from FEM simulations. Noise may be introduced following the comments in the code.

> [pinn_inverse_4param.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/Code/pinn_inverse_4param.py): PINN trained for the inverse problem to infer all of the gradient energy coefficients using labeled data from FEM simulations. Noise may be introduced following the comments in the code.

> [plot_experimental_data.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/code/plot_experimental_data.py): how to deal with the raw data from the experiment.

> [pinn_inverse_experiment_2param.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/code/pinn_inverse_experiment_2param.py): PINN trained for the inverse problem to infer two of the gradient energy coefficients using labeled data from an experiment.

> [plot_polarization_difference.py](https://github.com/whshangl/Open_PinnPhaseField/blob/main/code/plot_polarization_difference.py): One group of gradient energy coefficients is from a PINN, and the other group from literature. With the two groups of coefficients, we obtain two datasets of polarization in equilibrium via FEM simulations. Compare the two datasets with the experimental data.

