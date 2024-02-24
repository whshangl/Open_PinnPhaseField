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
time_length = 0.1
thickness = 0.2 


L_norm = domain_length/2
t_norm = time_length
Z_norm = thickness/2


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
G11 = 0.27689
G12 = 0
G44 = 0.13840
G44_= 0.13840


### governing equations
def pde(X, Y):
    """
    Expresses the PDE of the phase-field model. 
    Argument X to pde(X,Y) is the input, where X[:, 0] is x-coordinate, X[:,1] is y-coordination, X[:,2] is y-coordination, and X[:,3] is t(time)-coordinate.
    Argument Y to pde(X,Y) is the output, with 5 variables u1, u2, u3, phi, P1, P2, P3, as shown below.
    """
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    u3  = Y[:, 2:3]   ## displacement in 3-direction
    phi = Y[:, 3:4]   ## electric potential 
    P1  = Y[:, 4:5]   ## polarization in 1-direction
    P2  = Y[:, 5:6]   ## polarization in 2-direction
    P3  = Y[:, 6:7]   ## polarization in 3-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    u3_x  = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{u3}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 5, j = 0)   ## \frac{\partial{P2}}{\partial{x}}
    P3_x  = dde.grad.jacobian(Y, X, i = 6, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    u3_y  = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{u3}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 5, j = 1)   ## \frac{\partial{P2}}{\partial{y}}
    P3_y  = dde.grad.jacobian(Y, X, i = 6, j = 1)   ## \frac{\partial{P3}}{\partial{y}}

    u1_z  = dde.grad.jacobian(Y, X, i = 0, j = 2)   ## \frac{\partial{u1}}{\partial{y}}
    u2_z  = dde.grad.jacobian(Y, X, i = 1, j = 2)   ## \frac{\partial{u2}}{\partial{y}}
    u3_z  = dde.grad.jacobian(Y, X, i = 2, j = 2)   ## \frac{\partial{u3}}{\partial{y}}
    phi_z = dde.grad.jacobian(Y, X, i = 3, j = 2)   ## \frac{\partial{phi}}{\partial{y}}
    P1_z  = dde.grad.jacobian(Y, X, i = 4, j = 2)   ## \frac{\partial{P1}}{\partial{y}}
    P2_z  = dde.grad.jacobian(Y, X, i = 5, j = 2)   ## \frac{\partial{P2}}{\partial{y}}
    P3_z  = dde.grad.jacobian(Y, X, i = 6, j = 2)   ## \frac{\partial{P3}}{\partial{y}}

    u1_xx   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 0)  ## \frac{\partial^2{u1}}{\partial{x}^2}
    u2_xx   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 0)  ## \frac{\partial^2{u2}}{\partial{x}^2}
    u3_xx   = dde.grad.hessian(Y, X, component= 2, i = 0, j = 0)  ## \frac{\partial^2{u3}}{\partial{x}^2}
    phi_xx  = dde.grad.hessian(Y, X, component= 3, i = 0, j = 0)  ## \frac{\partial^2{phi}}{\partial{x}^2}
    P1_xx   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 0)  ## \frac{\partial^2{P1}}{\partial{x}^2}
    P2_xx   = dde.grad.hessian(Y, X, component= 5, i = 0, j = 0)  ## \frac{\partial^2{P2}}{\partial{x}^2}
    P3_xx   = dde.grad.hessian(Y, X, component= 6, i = 0, j = 0)  ## \frac{\partial^2{P3}}{\partial{x}^2}


    u1_yy   = dde.grad.hessian(Y, X, component= 0, i = 1, j = 1)  ## \frac{\partial^2{u1}}{\partial{y}^2}
    u2_yy   = dde.grad.hessian(Y, X, component= 1, i = 1, j = 1)  ## \frac{\partial^2{u2}}{\partial{y}^2}
    u3_yy   = dde.grad.hessian(Y, X, component= 2, i = 1, j = 1)  ## \frac{\partial^2{u3}}{\partial{y}^2}
    phi_yy  = dde.grad.hessian(Y, X, component= 3, i = 1, j = 1)  ## \frac{\partial^2{phi}}{\partial{y}^2}
    P1_yy   = dde.grad.hessian(Y, X, component= 4, i = 1, j = 1)  ## \frac{\partial^2{P1}}{\partial{y}^2}
    P2_yy   = dde.grad.hessian(Y, X, component= 5, i = 1, j = 1)  ## \frac{\partial^2{P2}}{\partial{y}^2}
    P3_yy   = dde.grad.hessian(Y, X, component= 6, i = 1, j = 1)  ## \frac{\partial^2{P3}}{\partial{y}^2}

    u1_zz   = dde.grad.hessian(Y, X, component= 0, i = 2, j = 2)  ## \frac{\partial^2{u1}}{\partial{z}^2}
    u2_zz   = dde.grad.hessian(Y, X, component= 1, i = 2, j = 2)  ## \frac{\partial^2{u2}}{\partial{z}^2}
    u3_zz   = dde.grad.hessian(Y, X, component= 2, i = 2, j = 2)  ## \frac{\partial^2{u3}}{\partial{z}^2}
    phi_zz  = dde.grad.hessian(Y, X, component= 3, i = 2, j = 2)  ## \frac{\partial^2{phi}}{\partial{z}^2}
    P1_zz   = dde.grad.hessian(Y, X, component= 4, i = 2, j = 2)  ## \frac{\partial^2{P1}}{\partial{z}^2}
    P2_zz   = dde.grad.hessian(Y, X, component= 5, i = 2, j = 2)  ## \frac{\partial^2{P2}}{\partial{z}^2}
    P3_zz   = dde.grad.hessian(Y, X, component= 6, i = 2, j = 2)  ## \frac{\partial^2{P3}}{\partial{z}^2}


    u1_xy   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 1)  ## \frac{\partial^2{u1}}{\partial{x}\partial{y}}
    u2_xy   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 1)  ## \frac{\partial^2{u2}}{\partial{x}\partial{y}}
    u3_xy   = dde.grad.hessian(Y, X, component= 2, i = 0, j = 1)  ## \frac{\partial^2{u3}}{\partial{x}\partial{y}}
    phi_xy  = dde.grad.hessian(Y, X, component= 3, i = 0, j = 1)  ## \frac{\partial^2{phi}}{\partial{x}\partial{y}}
    P1_xy   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 1)  ## \frac{\partial^2{P1}}{\partial{x}\partial{y}}
    P2_xy   = dde.grad.hessian(Y, X, component= 5, i = 0, j = 1)  ## \frac{\partial^2{P2}}{\partial{x}\partial{y}}
    P3_xy   = dde.grad.hessian(Y, X, component= 6, i = 0, j = 1)  ## \frac{\partial^2{P3}}{\partial{x}\partial{y}}


    u1_xz   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 2)  ## \frac{\partial^2{u1}}{\partial{x}\partial{z}}
    u2_xz   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 2)  ## \frac{\partial^2{u2}}{\partial{x}\partial{z}}
    u3_xz   = dde.grad.hessian(Y, X, component= 2, i = 0, j = 2)  ## \frac{\partial^2{u3}}{\partial{x}\partial{z}}
    phi_xz  = dde.grad.hessian(Y, X, component= 3, i = 0, j = 2)  ## \frac{\partial^2{phi}}{\partial{x}\partial{z}}
    P1_xz   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 2)  ## \frac{\partial^2{P1}}{\partial{x}\partial{z}}
    P2_xz   = dde.grad.hessian(Y, X, component= 5, i = 0, j = 2)  ## \frac{\partial^2{P2}}{\partial{x}\partial{z}}
    P3_xz   = dde.grad.hessian(Y, X, component= 6, i = 0, j = 2)  ## \frac{\partial^2{P3}}{\partial{x}\partial{z}}

    u1_yz   = dde.grad.hessian(Y, X, component= 0, i = 1, j = 2)  ## \frac{\partial^2{u1}}{\partial{x}\partial{z}}
    u2_yz   = dde.grad.hessian(Y, X, component= 1, i = 1, j = 2)  ## \frac{\partial^2{u2}}{\partial{x}\partial{z}}
    u3_yz   = dde.grad.hessian(Y, X, component= 2, i = 1, j = 2)  ## \frac{\partial^2{u3}}{\partial{x}\partial{z}}
    phi_yz  = dde.grad.hessian(Y, X, component= 3, i = 1, j = 2)  ## \frac{\partial^2{phi}}{\partial{x}\partial{z}}
    P1_yz   = dde.grad.hessian(Y, X, component= 4, i = 1, j = 2)  ## \frac{\partial^2{P1}}{\partial{x}\partial{z}}
    P2_yz   = dde.grad.hessian(Y, X, component= 5, i = 1, j = 2)  ## \frac{\partial^2{P2}}{\partial{x}\partial{z}}
    P3_yz   = dde.grad.hessian(Y, X, component= 6, i = 1, j = 2)  ## \frac{\partial^2{P3}}{\partial{x}\partial{z}}


    P1_t = dde.grad.jacobian(Y, X,  i = 4, j = 3)  ## \dot{P1}
    P2_t = dde.grad.jacobian(Y, X,  i = 5, j = 3)  ## \dot{P2}
    P3_t = dde.grad.jacobian(Y, X,  i = 6, j = 3)  ## \dot{P2}
    
    ###############################################################
    ### div(sigma) = 0 related expressions
    epsilon11_ = u1_x/L_norm   
    epsilon22_ = u2_y/L_norm
    epsilon33_ = u3_z/Z_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm
    epsilon23_ = 0.5 * (u2_z/Z_norm + u3_y/L_norm)
    epsilon13_ = 0.5 * (u1_z/Z_norm + u3_x/L_norm)

    epsilon11_x_ = u1_xx/L_norm/L_norm
    epsilon11_y_ = u1_xy/L_norm/L_norm
    epsilon11_z_ = u1_xz/L_norm/Z_norm

    epsilon22_x_ = u2_xy/L_norm/L_norm
    epsilon22_y_ = u2_yy/L_norm/L_norm
    epsilon22_z_ = u2_yz/L_norm/Z_norm

    epsilon33_x_ = u3_xz/L_norm/Z_norm
    epsilon33_y_ = u3_yz/L_norm/Z_norm
    epsilon33_z_ = u3_zz/Z_norm/Z_norm

    epsilon12_x_ = 0.5 * (u1_xy + u2_xx)/L_norm/L_norm
    epsilon12_y_ = 0.5 * (u1_yy + u2_xy)/L_norm/L_norm
    epsilon12_z_ = 0.5 * (u1_yz + u2_xz)/L_norm/Z_norm

    epsilon23_x_ = 0.5 * (u2_xz/Z_norm + u3_xy/L_norm)/L_norm
    epsilon23_y_ = 0.5 * (u2_yz/Z_norm + u3_yy/L_norm)/L_norm
    epsilon23_z_ = 0.5 * (u2_zz/Z_norm + u3_yz/L_norm)/Z_norm

    epsilon13_x_ = 0.5 * (u1_xz/Z_norm + u3_xx/L_norm)/L_norm
    epsilon13_y_ = 0.5 * (u1_yz/Z_norm + u3_xy/L_norm)/L_norm
    epsilon13_z_ = 0.5 * (u1_zz/Z_norm + u3_xz/L_norm)/Z_norm

    
    P1_x_ = P1_x/L_norm
    P2_x_ = P2_x/L_norm
    P3_x_ = P3_x/L_norm

    P1_y_ = P1_y/L_norm
    P2_y_ = P2_y/L_norm
    P3_y_ = P3_y/L_norm

    P1_z_ = P1_z/Z_norm
    P2_z_ = P2_z/Z_norm
    P3_z_ = P3_z/Z_norm
    
    P1_xx_ = P1_xx/L_norm/L_norm
    P1_yy_ = P1_yy/L_norm/L_norm
    P1_zz_ = P1_zz/Z_norm/Z_norm

    P2_xx_ = P2_xx/L_norm/L_norm
    P2_yy_ = P2_yy/L_norm/L_norm
    P2_zz_ = P2_zz/Z_norm/Z_norm

    P3_xx_ = P3_xx/L_norm/L_norm
    P3_yy_ = P3_yy/L_norm/L_norm
    P3_zz_ = P3_zz/Z_norm/Z_norm

    P1_xy_ = P1_xy/L_norm/L_norm
    P2_xy_ = P2_xy/L_norm/L_norm
    P3_xy_ = P3_xy/L_norm/L_norm

    P1_xz_ = P1_xz/L_norm/Z_norm
    P2_xz_ = P2_xz/L_norm/Z_norm
    P3_xz_ = P3_xz/L_norm/Z_norm

    P1_yz_ = P1_yz/L_norm/Z_norm
    P2_yz_ = P2_yz/L_norm/Z_norm
    P3_yz_ = P3_yz/L_norm/Z_norm
    
    P1_t_ = P1_t/t_norm
    P2_t_ = P2_t/t_norm
    P3_t_ = P3_t/t_norm

    ### stress
    sigma11 = c11 * epsilon11_ + c12 * (epsilon22_ + epsilon33_) - q11 * P1 * P1 - q12 * (P2 * P2 + P3 * P3)
    sigma22 = c11 * epsilon22_ + c12 * (epsilon11_ + epsilon33_) - q11 * P2 * P2 - q12 * (P1 * P1 + P3 * P3)
    sigma33 = c11 * epsilon33_ + c12 * (epsilon11_ + epsilon22_) - q11 * P3 * P3 - q12 * (P1 * P1 + P2 * P2)

    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    sigma13 = 2 * c44 * epsilon13_ - q44 * P1 * P3
    sigma23 = 2 * c44 * epsilon23_ - q44 * P2 * P3
    
    ### divergence of stress
    sigma11_x = c11 * epsilon11_x_ + c12 * (epsilon22_x_ + epsilon33_x_) - 2 * q11 * P1 * P1_x_ - 2 * q12 * (P2 * P2_x_ + P3 * P3_x_)
    sigma12_y = 2 * c44 * epsilon12_y_ - q44 * P2 * P1_y_ - q44 * P1 * P2_y_
    sigma13_z = 2 * c44 * epsilon13_z_ - q44 * P3 * P1_z_ - q44 * P1 * P3_z_

    sigma12_x = 2 * c44 * epsilon12_x_ - q44 * P2 * P1_x_ - q44 * P1 * P2_x_
    sigma22_y = c11 * epsilon22_y_ +  c12 * (epsilon11_y_ + epsilon33_y_) - 2 * q11 * P2 * P2_y_ - 2 * q12 * (P1 * P1_y_ + P3 * P3_y_)
    sigma23_z = 2 * c44 * epsilon23_z_ - q44 * P2 * P3_z_ - q44 * P2_z_ * P3

    sigma13_x = 2 * c44 * epsilon13_x_ - q44 * P3 * P1_x_ - q44 * P1 * P3_x_
    sigma23_y = 2 * c44 * epsilon23_y_ - q44 * P2 * P3_y_ - q44 * P2_y_ * P3
    sigma33_z = c11 * epsilon33_z_ +  c12 * (epsilon11_z_ + epsilon22_z_) - 2 * q11 * P3 * P3_z_ - 2 * q12 * (P1 * P1_z_ + P2 * P2_z_)

    ###############################################################
    ### div(D) = 0 related expressions
    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm
    E3_ = -phi_z/Z_norm

    E1_x_ = -phi_xx/L_norm/L_norm
    E2_y_ = -phi_yy/L_norm/L_norm
    E3_z_ = -phi_zz/Z_norm/Z_norm

    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2
    D3 = kappa * E3_ + P3

    ### divergence of electric displacement
    D1_x = kappa * E1_x_ + P1_x_
    D2_y = kappa * E2_y_ + P2_y_
    D3_z = kappa * E3_z_ + P3_z_

    ###############################################################
    ### TDGL equation related expressions
    ### h_P1 = \frac{\partial{h}}{\partial{P1}}
    h_P1 = + 2 * a1 * P1 + 4 * a11 * (P1**3) + 2 * a12 * P1 * (P2**2 + P3**2) \
           + 6 * a111 * (P1**5) + 4 * a112 * (P1**3) * (P2**2 + P3**2) \
           + 2 * a112 * P1 * (P2**4 + P3**4) + 2 * a123 * P1 * (P2**2) * (P3**2) \
           - 2 * q11 * epsilon11_ * P1 - 2 * q12 * P1 * (epsilon22_ + epsilon33_) \
           - 2 * q44 * (epsilon12_ * P2 + epsilon13_ * P3) \
           - E1_ 
           

    ### h_P2 = \frac{\partial{h}}{\partial{P2}}
    h_P2 = + 2 * a1 * P2 + 4 * a11 * (P2**3) + 2 * a12 * P2 * (P1**2 + P3**2) \
           + 6 * a111 * (P2**5) + 4 * a112 * (P2**3) * (P1**2 + P3**2) \
           + 2 * a112 * P2 * (P1**4 + P3**4) + 2 * a123 * P2 * (P1**2) * (P3**2) \
           - 2 * q11 * epsilon22_ * P2 - 2 * q12 * P2 * (epsilon11_ + epsilon33_) \
           - 2 * q44 * (epsilon12_ * P1 + epsilon23_ * P3) \
           - E2_ 
    
    ### h_P3 = \frac{\partial{h}}{\partial{P3}}
    h_P3 = + 2 * a1 * P3 + 4 * a11 * (P3**3) + 2 * a12 * P2 * (P2**2 + P3**2) \
           + 6 * a111 * (P3**5) + 4 * a112 * (P3**3) * (P1**2 + P2**2) \
           + 2 * a112 * P3 * (P1**4 + P2**4) + 2 * a123 * P3 * (P1**2) * (P2**2) \
           - 2 * q11 * epsilon33_ * P3 - 2 * q12 * P3 * (epsilon11_ + epsilon22_) \
           - 2 * q44 * (epsilon13_ * P1 + epsilon23_ * P2) \
           - E3_ 
        
    
    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    chi11 = G11 * P1_x_ + G12 * (P2_y_ + P3_z_)
    chi12 = G44 * (P1_y_ + P2_x_) + G44_ * (P1_y_ - P2_x_)
    chi13 = G44 * (P1_z_ + P3_x_) + G44_ * (P1_z_ - P3_x_)

    chi21 = G44 * (P1_y_ + P2_x_) + G44_ * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * (P1_x_ + P3_z_)
    chi23 = G44 * (P2_z_ + P3_y_) + G44_ * (P2_z_ - P3_y_)

    chi31 = G44 * (P1_z_ + P3_x_) + G44_ * (P3_x_ - P1_z_)
    chi32 = G44 * (P2_z_ + P3_y_) + G44_ * (P3_y_ - P2_z_)
    chi33 = G11 * P3_z_ + G12 * (P2_y_ + P1_x_)

    
    ### divergence of chi_{ij}
    chi11_x = G11 * P1_xx_ + G12 * (P2_xy_ + P3_xz_)
    chi12_y = G44 * (P1_yy_ + P2_xy_) + G44_ * (P1_yy_ - P2_xy_)
    chi13_z = G44 * (P1_zz_ + P3_xz_) + G44_ * (P1_zz_ - P3_xz_)

    chi21_x = G44 * (P1_xy_ + P2_xx_) + G44_ * (P2_xx_ - P1_xy_)
    chi22_y = G11 * P2_yy_ + G12 * (P1_xy_ + P3_yz_)
    chi23_z = G11 * (P2_zz_ + P3_yz_) + G44_ * (P2_zz_ - P3_yz_)

    chi31_x = G44 * (P1_xz_ + P3_xx_) + G44_ * (P3_xx_ - P1_xz_)
    chi32_y = G44 * (P2_yz_ + P3_yy_) + G44_ * (P3_yy_ - P2_yz_)
    chi33_z = G11 * P3_zz_ + G12 * (P2_yz_ + P1_xz_)

    ### divergence of {chi_{ij}}_{2*2}
    div_P1 = chi11_x + chi12_y + chi13_z
    div_P2 = chi21_x + chi22_y + chi23_z
    div_P3 = chi31_x + chi32_y + chi33_z

    ###############################################################
    ### balance equations
    balance_mechanic_1 = sigma11_x + sigma12_y + sigma13_z
    balance_mechanic_2 = sigma12_x + sigma22_y + sigma23_z
    balance_mechanic_3 = sigma13_x + sigma23_y + sigma33_z

    balance_electric = D1_x + D2_y + D3_z

    TDGL_1 = P1_t_ + h_P1 - div_P1
    TDGL_2 = P2_t_ + h_P2 - div_P2
    TDGL_3 = P3_t_ + h_P3 - div_P3
    
    return [balance_mechanic_1, balance_mechanic_2, balance_mechanic_3, balance_electric, TDGL_1, TDGL_2, TDGL_3]



### Computational geometry:
spatial_domain = dde.geometry.Cuboid(xmin=[-1*domain_length/2/L_norm, -1*domain_length/2/L_norm, -1*thickness/2/Z_norm], xmax=[domain_length/2/L_norm, domain_length/2/L_norm, thickness/2/Z_norm])
temporal_domain = dde.geometry.TimeDomain(0, time_length/t_norm)
geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)


def boundary_left_right(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm))

def boundary_bottom_top(X, on_boundary):
    return on_boundary and (np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))

def boundary_front_back(X, on_boundary):
    return on_boundary and (np.isclose(X[2], -1*thickness/2/Z_norm) or np.isclose(X[2], thickness/2/Z_norm))

def boundary_XY(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm)) or (np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))

def boundary_XZ(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm)) or (np.isclose(X[2], -1*thickness/2/Z_norm) or np.isclose(X[2], thickness/2/Z_norm))

def boundary_YZ(X, on_boundary):
    return on_boundary and (np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm)) or (np.isclose(X[2], -1*thickness/2/Z_norm) or np.isclose(X[2], thickness/2/Z_norm))

def boundary_flux(X,Y):
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    u3  = Y[:, 2:3]   ## displacement in 3-direction
    phi = Y[:, 3:4]   ## electric potential 
    P1  = Y[:, 4:5]   ## polarization in 1-direction
    P2  = Y[:, 5:6]   ## polarization in 2-direction
    P3  = Y[:, 6:7]   ## polarization in 3-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    u3_x  = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{u3}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 5, j = 0)   ## \frac{\partial{P2}}{\partial{x}}
    P3_x  = dde.grad.jacobian(Y, X, i = 6, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    u3_y  = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{u3}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 5, j = 1)   ## \frac{\partial{P2}}{\partial{y}}
    P3_y  = dde.grad.jacobian(Y, X, i = 6, j = 1)   ## \frac{\partial{P3}}{\partial{y}}

    u1_z  = dde.grad.jacobian(Y, X, i = 0, j = 2)   ## \frac{\partial{u1}}{\partial{y}}
    u2_z  = dde.grad.jacobian(Y, X, i = 1, j = 2)   ## \frac{\partial{u2}}{\partial{y}}
    u3_z  = dde.grad.jacobian(Y, X, i = 2, j = 2)   ## \frac{\partial{u3}}{\partial{y}}
    phi_z = dde.grad.jacobian(Y, X, i = 3, j = 2)   ## \frac{\partial{phi}}{\partial{y}}
    P1_z  = dde.grad.jacobian(Y, X, i = 4, j = 2)   ## \frac{\partial{P1}}{\partial{y}}
    P2_z  = dde.grad.jacobian(Y, X, i = 5, j = 2)   ## \frac{\partial{P2}}{\partial{y}}
    P3_z  = dde.grad.jacobian(Y, X, i = 6, j = 2)   ## \frac{\partial{P3}}{\partial{y}}

    
    ###############################################################
    ### div(sigma) = 0 related expressions
    epsilon11_ = u1_x/L_norm   
    epsilon22_ = u2_y/L_norm
    epsilon33_ = u3_z/Z_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm
    epsilon23_ = 0.5 * (u2_z/Z_norm + u3_y/L_norm)
    epsilon13_ = 0.5 * (u1_z/Z_norm + u3_x/L_norm)


    P1_x_ = P1_x/L_norm
    P2_x_ = P2_x/L_norm
    P3_x_ = P3_x/L_norm

    P1_y_ = P1_y/L_norm
    P2_y_ = P2_y/L_norm
    P3_y_ = P3_y/L_norm

    P1_z_ = P1_z/Z_norm
    P2_z_ = P2_z/Z_norm
    P3_z_ = P3_z/Z_norm
    

    ### stress
    sigma11 = c11 * epsilon11_ + c12 * (epsilon22_ + epsilon33_) - q11 * P1 * P1 - q12 * (P2 * P2 + P3 * P3)
    sigma22 = c11 * epsilon22_ + c12 * (epsilon11_ + epsilon33_) - q11 * P2 * P2 - q12 * (P1 * P1 + P3 * P3)
    sigma33 = c11 * epsilon33_ + c12 * (epsilon11_ + epsilon22_) - q11 * P3 * P3 - q12 * (P1 * P1 + P2 * P2)

    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    sigma13 = 2 * c44 * epsilon13_ - q44 * P1 * P3
    sigma23 = 2 * c44 * epsilon23_ - q44 * P2 * P3
    
    ###############################################################
    ### div(D) = 0 related expressions
    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm
    E3_ = -phi_z/Z_norm


    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2
    D3 = kappa * E3_ + P3

   
    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    chi11 = G11 * P1_x_ + G12 * (P2_y_ + P3_z_)
    chi12 = G44 * (P1_y_ + P2_x_) + G44_ * (P1_y_ - P2_x_)
    chi13 = G44 * (P1_z_ + P3_x_) + G44_ * (P1_z_ - P3_x_)

    chi21 = G44 * (P1_y_ + P2_x_) + G44_ * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * (P1_x_ + P3_z_)
    chi23 = G44 * (P2_z_ + P3_y_) + G44_ * (P2_z_ - P3_y_)

    chi31 = G44 * (P1_z_ + P3_x_) + G44_ * (P3_x_ - P1_z_)
    chi32 = G44 * (P2_z_ + P3_y_) + G44_ * (P3_y_ - P2_z_)
    chi33 = G11 * P3_z_ + G12 * (P2_y_ + P1_x_)


    return [sigma11, sigma22, sigma33, sigma12, sigma23, sigma13, D1, D2, D3, chi11, chi21, chi31, chi22, chi12, chi32, chi33, chi13, chi23]


#### boundary condition: sigma*n = 0 (traction free)
bc_LeftRight_traction_11 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[0], boundary_left_right)
bc_BottomTop_traction_22 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[1], boundary_bottom_top)
bc_FrontBack_traction_33 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[2], boundary_front_back)
bc_XY_traction_12        = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[3], boundary_XY)
bc_YZ_traction_23        = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[4], boundary_YZ)
bc_XZ_traction_13        = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[5], boundary_XZ)

### boundary condition: D*n = 0 (surface charge free)
bc_LeftRight_charge_1 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[6], boundary_left_right)
bc_BottomTop_charge_2 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[7], boundary_bottom_top)
bc_FrontBack_charge_3 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[8], boundary_front_back)

### boundary condition: surface gradient flux free
bc_LeftRight_gradient_11 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[9], boundary_left_right)
bc_LeftRight_gradient_21 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[10], boundary_left_right)
bc_LeftRight_gradient_31 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[11], boundary_left_right)

bc_BottomTop_gradient_22 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[12], boundary_bottom_top)
bc_BottomTop_gradient_12 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[13], boundary_bottom_top)
bc_BottomTop_gradient_32 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[14], boundary_bottom_top)

bc_FrontBack_gradient_33 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[15], boundary_bottom_top)
bc_FrontBack_gradient_13 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[16], boundary_bottom_top)
bc_FrontBack_gradient_23 = dde.icbc.OperatorBC(geomtime, lambda X, Y, _: boundary_flux(X,Y)[17], boundary_bottom_top)


bc_ic =  [
          bc_LeftRight_traction_11,  bc_BottomTop_traction_22,  bc_FrontBack_traction_33,
          bc_XY_traction_12,         bc_YZ_traction_23,         bc_XZ_traction_13,
          bc_LeftRight_charge_1,     bc_BottomTop_charge_2,     bc_FrontBack_charge_3,
          bc_LeftRight_gradient_11,  bc_LeftRight_gradient_21,  bc_LeftRight_gradient_31, 
          bc_BottomTop_gradient_22,  bc_BottomTop_gradient_12,  bc_BottomTop_gradient_32, 
          bc_FrontBack_gradient_33,  bc_FrontBack_gradient_13,  bc_FrontBack_gradient_23
         ]

data = dde.data.TimePDE(
    geomtime,
    pde,
    bc_ic,
    num_domain   = 20000,
    num_boundary = 6000,
    num_test     = 30000,
    train_distribution='Hammersley',
    anchors = None,
)

pde_resampler = dde.callbacks.PDEPointResampler(period=2000, pde_points=True, bc_points=False)

## network
nn_layer_size = [4] + [50] * 3 + [7]  
activation  = "tanh"
initializer = "Glorot normal"

net = dde.nn.FNN(nn_layer_size, activation, initializer)

def transform_func(X, Y):
    x, y, z, t = X[:,0:1], X[:,1:2], X[:,2:3], X[:,3:4] 
    u1, u2, u3, phi, P1, P2, P3 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4], Y[:, 4:5], Y[:, 5:6], Y[:, 6:7]

    P1_0 = x/2 * L_norm
    P2_0 = y/2 * L_norm
    P3_0 = z * Z_norm
    P1_new  =  P1 * t * t_norm * 0.1 + P1_0 
    P2_new  =  P2 * t * t_norm * 0.1 + P2_0
    P3_new  =  P3 * t * t_norm * 0.1 + P3_0

    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    u3_new = u3 * 1e-3

    phi_new = phi * 1e-1
    
    return torch.cat((u1_new, u2_new, u3_new, phi_new, P1_new, P2_new,  P3_new), dim =1)

net.apply_output_transform(transform_func)

model = dde.Model(data, net)

store_path = Path('./3D_PF/epoch_3D_PF')
store_path.mkdir(parents=True, exist_ok=True)

checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = store_path, 
    verbose  = 1,
    save_better_only=True,  
    period=2000,
    monitor = 'train loss'
)

loss_weights = [
                1,     1,     1,    10,   100,  100,  100,   ## 7 pdes
                100,   100,   100,                           ## 3 bcs of traction free
                100,   100,   100,                           ## 3 bcs of traction free
                100,   100,   100,                           ## 3 bcs of charge   free
                100,   100,   100,                           ## 3 bcs of gradient free 
                100,   100,   100,                           ## 3 bcs of gradient free 
                100,   100,   100                            ## 3 bcs of gradient free                          
                ] 


dde.optimizers.config.set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-08,
    maxiter=30000,
    maxfun=None,
    maxls=50,
)


###################################################################################
############################ train ############################
###################################################################################

begin_time = datetime.now()
print("Training starts from {}".format(begin_time))

model.compile("adam", lr=1e-3, loss = 'MSE', loss_weights = loss_weights)
losshistory, train_state = model.train(iterations = 10000, display_every=1000, model_save_path = store_path, callbacks=[checkpointer,pde_resampler])

model.compile("adam", lr=1e-4, loss = 'MSE', loss_weights = loss_weights)
losshistory, train_state = model.train(iterations = 20000, display_every=1000, model_save_path = store_path, callbacks=[checkpointer, pde_resampler])

model.compile("L-BFGS", loss = 'MSE', loss_weights = loss_weights)
losshistory, train_state = model.train(display_every=1000, model_save_path = store_path, callbacks=[checkpointer,pde_resampler])

end_time = datetime.now()
print("Training ends at {}".format(end_time))
print("Total time spent on training: {}".format(end_time - begin_time))

dde.saveplot(losshistory, train_state, issave=True, isplot= True, output_dir=store_path)

###################################################################################
############################ train ############################
###################################################################################



