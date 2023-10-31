"""Backend supported: pytorch"""
import deepxde as dde
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

dde.config.set_random_seed(1001)
dde.config.set_default_float('float32')

device = torch.device("cuda:0")
torch.set_default_device(device)

torch.cuda.empty_cache()

### Problem parameters:
domain_length = 4
time_length = 5

L_norm = domain_length/2
t_norm = time_length

# L_norm = 1 
# t_norm = 1

a1 = -0.1725
kappa = 0.5844
a11 = -0.072245
a12 = 0.75255
a111 = 0.25740
a112 = 0.63036
a123 = -3.6771
c11 = 174.57
c12 = 79.278
c44 = 111.11
q11 = 10.919
q12 = 0.4485
q44 = 7.1760
# G11 = 0.27689
G12 = 0
# G44 = 0.13840
# G44_= 0.13840

### parameters to be determined
G11 = dde.Variable(1.0)
G44 = dde.Variable(1.0)  ### assume G44_ = G44


### governing equations
def pde(X, Y):
    """
    Expresses the PDE of the phase-field model. 
    Argument X to pde(X,Y) is the input, where X[:, 0] is x-coordinate, X[:,1] is y-coordination, and X[:,2] is t(time)-coordinate.
    Argument Y to pde(X,Y) is the output, with 5 variables u1, u2, phi, P1, P2, as shown below.
    """
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    phi = Y[:, 2:3]   ## electric potential 
    P1  = Y[:, 3:4]   ## polarization in 1-direction
    P2  = Y[:, 4:5]   ## polarization in 2-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P2}}{\partial{y}}

    u1_xx   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 0)  ## \frac{\partial^2{u1}}{\partial{x}^2}
    u2_xx   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 0)  ## \frac{\partial^2{u2}}{\partial{x}^2}
    phi_xx  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 0)  ## \frac{\partial^2{phi}}{\partial{x}^2}
    P1_xx   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 0)  ## \frac{\partial^2{P1}}{\partial{x}^2}
    P2_xx   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 0)  ## \frac{\partial^2{P2}}{\partial{x}^2}

    u1_yy   = dde.grad.hessian(Y, X, component= 0, i = 1, j = 1)  ## \frac{\partial^2{u1}}{\partial{y}^2}
    u2_yy   = dde.grad.hessian(Y, X, component= 1, i = 1, j = 1)  ## \frac{\partial^2{u2}}{\partial{y}^2}
    phi_yy  = dde.grad.hessian(Y, X, component= 2, i = 1, j = 1)  ## \frac{\partial^2{phi}}{\partial{y}^2}
    P1_yy   = dde.grad.hessian(Y, X, component= 3, i = 1, j = 1)  ## \frac{\partial^2{P1}}{\partial{y}^2}
    P2_yy   = dde.grad.hessian(Y, X, component= 4, i = 1, j = 1)  ## \frac{\partial^2{P2}}{\partial{y}^2}

    u1_xy   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 1)  ## \frac{\partial^2{u1}}{\partial{x}\partial{y}}
    u2_xy   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 1)  ## \frac{\partial^2{u2}}{\partial{x}\partial{y}}
    phi_xy  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 1)  ## \frac{\partial^2{phi}}{\partial{x}\partial{y}}
    P1_xy   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 1)  ## \frac{\partial^2{P1}}{\partial{x}\partial{y}}
    P2_xy   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 1)  ## \frac{\partial^2{P2}}{\partial{x}\partial{y}}

    P1_t = dde.grad.jacobian(Y, X,  i = 3, j = 2)  ## \dot{P1}
    P2_t = dde.grad.jacobian(Y, X,  i = 4, j = 2)  ## \dot{P2}
    
    ###############################################################
    ### div(sigma) = 0 related expressions
    ### strain: plane strain assumption is used, i.e., epsilon33 = epsilon13 = epsilon23 = 0
    epsilon11_ = u1_x/L_norm   
    epsilon22_ = u2_y/L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm

    epsilon11_x_ = u1_xx/L_norm/L_norm
    epsilon11_y_ = u1_xy/L_norm/L_norm

    epsilon12_y_ = 0.5 * (u1_yy + u2_xy)/L_norm/L_norm
    epsilon12_x_ = 0.5 * (u1_xy + u2_xx)/L_norm/L_norm

    epsilon22_x_ = u2_xy/L_norm/L_norm
    epsilon22_y_ = u2_yy/L_norm/L_norm
    
    P1_x_ = P1_x/L_norm
    P2_x_ = P2_x/L_norm
    P1_y_ = P1_y/L_norm
    P2_y_ = P2_y/L_norm
    
    P1_xx_ = P1_xx/L_norm/L_norm
    P1_yy_ = P1_yy/L_norm/L_norm
    P1_xy_ = P1_xy/L_norm/L_norm
    
    P2_xx_ = P2_xx/L_norm/L_norm
    P2_yy_ = P2_yy/L_norm/L_norm
    P2_xy_ = P2_xy/L_norm/L_norm
    
    P1_t_ = P1_t/t_norm
    P2_t_ = P2_t/t_norm

    ### stress
    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    
    ### divergence of stress
    sigma11_x = c11 * epsilon11_x_ + c12 * epsilon22_x_ - 2 * q11 * P1 * P1_x_ - 2 * q12 * P2 * P2_x_
    sigma12_y = 2 * c44 * epsilon12_y_ - q44 * P2 * P1_y_ - q44 * P1 * P2_y_
    sigma12_x = 2 * c44 * epsilon12_x_ - q44 * P2 * P1_x_ - q44 * P1 * P2_x_
    sigma22_y = c11 * epsilon22_y_ +  c12 * epsilon11_y_ - 2 * q11 * P2 * P2_y_ - 2 * q12 * P1 * P1_y_
    
    ###############################################################
    ### div(D) = 0 related expressions
    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm

    E1_x_ = -phi_xx/L_norm/L_norm
    E2_y_ = -phi_yy/L_norm/L_norm

    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2

    ### divergence of electric displacement
    D1_x = kappa * E1_x_ + P1_x_
    D2_y = kappa * E2_y_ + P2_y_

    ###############################################################
    ### TDGL equation related expressions
    ### h_P1 = \frac{\partial{h}}{\partial{P1}}
    h_P1 = + 2 * a1 * P1 \
           + 4 * a11 * (P1**3)  \
           + 6 * a111 * (P1**5) \
           - 2 * q11 * epsilon11_ * P1 - 2 * q12 * P1 * epsilon22_ - 2 * q44 * epsilon12_ * P2 \
           - E1_ \
           + 2 * a12 * P1 * (P2**2) \
           + 4 * a112 * (P1**3) * (P2**2) + 2 * a112 * P1 * (P2**4) 

    
    ### h_P2 = \frac{\partial{h}}{\partial{P2}}
    h_P2 = + 2 * a1 * P2 \
           + 4 * a11 * (P2**3)  \
           + 6 * a111 * (P2**5) \
           - 2 * q11 * epsilon22_ * P2 - 2 * q12 * P2 * epsilon11_ - 2 * q44 * epsilon12_ * P1 \
           - E2_ \
           + 2 * a12 * P2 * (P1**2) \
           + 4 * a112 * (P2**3) * (P1**2) + 2 * a112 * P2 * (P1**4) 
    
    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    chi11 = G11 * P1_x_ + G12 * P2_y_
    chi12 = G44 * (P1_y_ + P2_x_) + G44 * (P1_y_ - P2_x_)
    chi21 = G44 * (P1_y_ + P2_x_) + G44 * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * P1_x_

    ### divergence of chi_{ij}
    chi11_x = G11 * P1_xx_ + G12 * P2_xy_
    chi12_y = G44 * (P1_yy_ + P2_xy_) + G44 * (P1_yy_ - P2_xy_)
    chi21_x = G44 * (P1_xy_ + P2_xx_) + G44 * (P2_xx_ - P1_xy_)
    chi22_y = G11 * P2_yy_ + G12 * P1_xy_

    ### divergence of {chi_{ij}}_{2*2}
    div_P1 = chi11_x + chi12_y 
    div_P2 = chi21_x + chi22_y

    ###############################################################
    ### balance equations
    balance_mechanic_1 = sigma11_x + sigma12_y
    balance_mechanic_2 = sigma12_x + sigma22_y

    balance_electric = D1_x + D2_y

    TDGL_1 = P1_t_ + h_P1 - div_P1
    TDGL_2 = P2_t_ + h_P2 - div_P2
    
    return [balance_mechanic_1, balance_mechanic_2, balance_electric, TDGL_1, TDGL_2]


### Computational geometry:
spatial_domain = dde.geometry.Rectangle(xmin=[-1*domain_length/2/L_norm, -1*domain_length/2/L_norm], xmax=[domain_length/2/L_norm, domain_length/2/L_norm])
temporal_domain = dde.geometry.TimeDomain(0, time_length/t_norm)
geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)


def boundary_left_right(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm))

def boundary_bottom_top(X, on_boundary):
    return on_boundary and (np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))

def boundary_all(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm) or np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))


def boundary_flux(X,Y):
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    phi = Y[:, 2:3]   ## electric potential 
    P1  = Y[:, 3:4]   ## polarization in 1-direction
    P2  = Y[:, 4:5]    ## polarization in 2-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P2}}{\partial{y}}


    ### strain: plane strain assumption is used, i.e., epsilon33 = epsilon13 = epsilon23 = 0
    epsilon11_ = u1_x/L_norm  
    epsilon22_ = u2_y/L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm
  
    ### stress
    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2

    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm

    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2

    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    
    P1_x_ = P1_x/L_norm
    P1_y_ = P1_y/L_norm
    
    P2_x_ = P2_x/L_norm
    P2_y_ = P2_y/L_norm
    
    chi11 = G11 * P1_x_ + G12 * P2_y_
    chi12 = G44 * (P1_y_ + P2_x_) + G44 * (P1_y_ - P2_x_)
    chi21 = G44 * (P1_y_ + P2_x_) + G44 * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * P1_x_

    return [sigma11, sigma22, sigma12, D1, D2, chi11, chi22, chi12, chi21]


#### boundary condition: sigma*n = 0 (traction free)
bc_LeftRight_traction_11 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[0], boundary_left_right)
bc_BottomTop_traction_22 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[1], boundary_bottom_top)
bc_All_traction_12       = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[2], boundary_all)

### boundary condition: D*n = 0 (surface charge free)
bc_LeftRight_charge_1 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[3], boundary_left_right)
bc_BottomTop_charge_2 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[4], boundary_bottom_top)

### boundary condition: surface gradient flux free
bc_LeftRight_gradient_11 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[5], boundary_left_right)
bc_LeftRight_gradient_21 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[8], boundary_left_right)

bc_BottomTop_gradient_22 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[6], boundary_bottom_top)
bc_BottomTop_gradient_12 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[7], boundary_bottom_top)


#### observation data for the steady domain
#### observation data for the steady domain
EXP_data = np.load('./experimental_data/experimental_data_domain.npy')

xy_observ = EXP_data[:, 0:2]
xyt_observ = np.hstack((xy_observ, time_length*np.ones([xy_observ.shape[0],1])))

norm_array = np.array([1/L_norm, 1/L_norm, 1/t_norm])

X_observ = xyt_observ * norm_array


dx_observe = EXP_data[:, 2:3]
dy_observe = EXP_data[:, 3:4]


#### If assuming the maximum value of polarization obtained from FEM links to the maximum value of experimental data ####
coef = 0.91
#### If assuming the maximum value of polarization obtained from FEM links to the maximum value of experimental data ####

P1_observ = dx_observe * coef
P2_observ = dy_observe * coef


##### add artificial noise
np.random.seed(123)
noise1 = np.random.normal(1, 0, P1_observ.shape[0]).reshape(P1_observ.shape)   ### no noise
np.random.seed(1234)
noise2 = np.random.normal(1, 0, P1_observ.shape[0]).reshape(P2_observ.shape)   ### no noise


P1_observ_noise = np.multiply(np.array(P1_observ), np.array(noise1))
P2_observ_noise = np.multiply(np.array(P2_observ), np.array(noise2))
#### add artificial noise

observ_bc_P1 = dde.icbc.PointSetBC(X_observ, P1_observ_noise, component=3)
observ_bc_P2 = dde.icbc.PointSetBC(X_observ, P2_observ_noise, component=4)


####### initial data manufactured from the above observation data ########

xy_init = EXP_data[:, 0:2]
xyt_init = np.hstack((xy_init, np.zeros((xy_init.shape[0],1))))

X_init = xyt_init * norm_array

P1_init = P1_observ_noise*0.95
P2_init = P2_observ_noise*0.95

ic_P1 = dde.icbc.PointSetBC(X_init, P1_init, component=3)
ic_P2 = dde.icbc.PointSetBC(X_init, P2_init, component=4)


bc_ic =  [ic_P1, ic_P2,
          bc_LeftRight_traction_11,  bc_BottomTop_traction_22,  bc_All_traction_12,
          bc_LeftRight_charge_1,     bc_BottomTop_charge_2,
          bc_LeftRight_gradient_11,  bc_LeftRight_gradient_21,  bc_BottomTop_gradient_22,  bc_BottomTop_gradient_12,
          observ_bc_P1, observ_bc_P2
         ]

anchor_points = np.vstack((X_observ, X_init))


data = dde.data.TimePDE(
    geomtime,
    pde,
    bc_ic,
    num_domain   = 20000,
    num_boundary = 4000,
    num_test     = 50000,
    train_distribution='Hammersley',
    anchors = anchor_points,
)

pde_resampler = dde.callbacks.PDEPointResampler(period=2000, pde_points=True, bc_points=False)


store_path = Path('./Inverse_experiment/epoch')
store_path.mkdir(parents=True, exist_ok=True)


fname_var = "./Inverse_experiment/varibles_2param.txt"
variable = dde.callbacks.VariableValue([G11, G44], period=200, filename=fname_var, precision=4)

## network
nn_layer_size = [3] + [100] * 3 + [5]  
activation  = "tanh"
initializer = "Glorot normal"

net = dde.nn.FNN(nn_layer_size, activation, initializer)

def transform_func(X, Y):
    x,y,t = X[:,0:1], X[:,1:2], X[:,2:3] 
    u1, u2, phi, P1, P2 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4], Y[:, 4:5]

    # P1_0 = torch.cos((np.pi*x/2) * L_norm)*torch.sin((np.pi*y/4) * L_norm)
    # P2_0 = torch.cos((np.pi*x/2) * L_norm)*torch.sin((np.pi*y/4) * L_norm)
    # P1_new  =  P1 * t * t_norm * 0.1 + P1_0 
    # P2_new  =  P2 * t * t_norm * 0.1 + P2_0


    # u1_new = (16 - x*x*y*y) * u1 * 1e-3
    # u2_new = (16 - x*x*y*y) * u2 * 1e-3
    # phi_new = (x*x + y*y) * phi * 2e-2

    #### at left bottom corner: u1 = u2 = phi = 0 
    # u1_new = ((x+1*domain_length/2/L_norm)+(y+1*domain_length/2/L_norm))*u1*1e-2
    # u2_new = ((x+1*domain_length/2/L_norm)+(y+1*domain_length/2/L_norm))*u2*1e-2
    # phi_new = ((x+1*domain_length/2/L_norm)+(y+1*domain_length/2/L_norm))*phi*1e-2
    
    P1_new  =  P1 * 0.1  
    P2_new  =  P2 * 0.1 
    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    phi_new = phi * 0.1
    
    
    return torch.cat((u1_new, u2_new, phi_new, P1_new, P2_new), dim =1)

net.apply_output_transform(transform_func)

model = dde.Model(data, net)



checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = store_path, 
    verbose  = 1,
    save_better_only=True,  
    period=1000,
    monitor = 'train loss'
)

loss_weights = [
                1,     1,     10,   100,   100,              ## 5 pdes
                1,     1,                                    ## 2 ics of P1, P2
                100,   100,   100,                           ## 3 bcs of traction free
                100,   100,                                  ## 2 bcs of charge free
                100,   100,   100,  100,                     ## 4 bcs of gradient free
                1,     1                                     ## observations
                ] 


dde.optimizers.config.set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-08,
    maxiter=40000,
    maxfun=None,
    maxls=50,
)


###################################################################################
############################ train ############################
###################################################################################
# early_stopping = dde.callbacks.EarlyStopping(min_delta=2e-5, patience = 5000)

begin_time = datetime.now()
print("Training starts from {}".format(begin_time))


model.compile("adam", lr=1e-4, loss = 'MSE', loss_weights = loss_weights, external_trainable_variables=[G11, G44])
losshistory, train_state = model.train(iterations = 30000, display_every=1000, model_save_path = store_path, callbacks=[checkpointer, pde_resampler, variable])

model.compile("adam", lr=1e-5, loss = 'MSE', loss_weights = loss_weights, external_trainable_variables=[G11, G44])
losshistory, train_state = model.train(iterations = 30000, display_every=1000, model_save_path = store_path, callbacks=[checkpointer, pde_resampler, variable])

model.compile("L-BFGS", loss = 'MSE', loss_weights = loss_weights, external_trainable_variables=[G11, G44])
losshistory, train_state = model.train(display_every=1000, model_save_path = store_path, callbacks=[checkpointer,pde_resampler, variable])

end_time = datetime.now()
print("Training ends at {}".format(end_time))
print("Total time spent on training: {}".format(end_time - begin_time))

dde.saveplot(losshistory, train_state, issave=True, isplot= True, output_dir = store_path)

#################################################################################
########################### train ############################
##################################################################################

