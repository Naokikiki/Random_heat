#!/usr/bin/env python
# coding: utf-8

# In[5]:


from __future__ import print_function
from fenics import *
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
from DLR import DLR2


# In[39]:


class Multi_Level_DLR:
    def __init__(self,level,dt,n,U,Y,a,a_0,a_sto,R ,sample_size = 50,mesh_type='2D'):
        self.level = level
        self.dt = dt  # time step
        self.n = n  # number of mesh * level
        self.h = 1 / np.array(n) # element size
        self.R = R  # rank of DLR * level
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh = [UnitIntervalMesh(self.n[i]) for i in range(self.level)]
        elif mesh_type == '2D':
            self.mesh = [UnitSquareMesh(self.n[i], self.n[i]) for i in range(self.level)]
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V = [FunctionSpace(self.mesh[i], 'P', 1) for i in range(self.level)]
       
        ## Define boundary condition
        self.bc = [DirichletBC(self.V[i], Constant(0), 'on_boundary') for i in range(self.level)]

        
        ## Initialize functions 
       
         # stochatic basis functions        
        self.Y = Y #(R,sample_size) * level
        for level in range(self.level):
            self.Y[level] = np.array(self.Y[level])
        self.Y_n = self.Y
        
        # deterministic basis functions
        self.U = [[interpolate(u_i, self.V[j]) for u_i in U[j]] for j in range(self.level)]#  (R,V_h) *level
        self.U_n =  [[interpolate(u_i, self.V[j]) for u_i in U[j]] for j in range(self.level)]
        
        ## set coefficient
        self.a_0 = a_0
        self.a_sto = a_sto
        self.a = a 
        

        #M_i,j = <U_i,U_j>
        self.matrix = [np.zeros((self.R[i],self.R[i])) for i in range(self.level)] 

        # Mass matrix
        self.mass_matrix = [self._assemble_mass_matrix(self.V[i]) for i in range(self.level)]
        
        #initial reorthogonalization
        self.reorthogonalize()
        
        # assign
        for level in range(self.level):
            for j in range(self.R[level]):
                self.U_n[level][j].assign(self.U[level][j])
        self.Y_n = self.Y
       
        ##for plot
        self.timelist = []
        self.energylist = []
        self.L2list= []

     # calculate Mass matrix
    def _assemble_mass_matrix(self, V):
            tri = TrialFunction(V)
            test = TestFunction(V)
            integral = tri * test * dx
            A = assemble(integral)
            return as_backend_type(A).mat().getValues(range(A.size(0)), range(A.size(1)))
   

    # calculate quadratures, M_i,j = <U_i,U_j>
    def matrix_calculate(self,level):
        v2d = vertex_to_dof_map(self.V[level])
        for i in range(self.R[level]):
            for j in range(i, self.R[level]):
                dof_i = self.U[level][i].vector()[v2d]
                dof_j = self.U[level][j].vector()[v2d]
                value = np.dot(dof_i, np.dot(self.mass_matrix[level], dof_j))
                self.matrix[level][i][j] = value
                self.matrix[level][j][i] = value

    ## subfunctions for calculating dinamics

    def _build_function(self, i,level):
        func = Constant(0)
        for j in range(self.R[level]):
            func += self.U_n[level][j] * Constant(self.Y_n[level][j][i])
        return func

    def E_a_grad_u_Y(self,a,Y,level):
        grad_list = [grad(self._build_function(i,level)) for i in range(self.sample_size)]
        ans = a[0]* grad_list[0]* Y[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i]* Y[i])
        ans /= self.sample_size
        return ans
    
    

    def a_grad_u_grad_U(self,a,level):
        ans = np.zeros((self.R[level],self.sample_size)) 
        u = [self._build_function(i,level) for i in range(self.sample_size)]
        for i in range(self.R[level]):
            for j in range(self.sample_size):
                ans[i][j] = assemble(dot(a[j] *grad(u[j]),grad(self.U[level][i])) * dx)        
        return ans
    
    
                             
    def orthogonal_projection(self,v,level):
        ortho = np.inner(v,self.Y[level][0]) / self.sample_size * self.Y[level][0]
        for i in range(1,self.R[level]):
            ortho += np.inner(v,self.Y[level][i]) / self.sample_size * self.Y[level][i]
        return v - ortho
    
    

    def reorthogonalize(self):
        for level in range(self.level):
            Q, _ = np.linalg.qr(np.transpose(self.Y[level]))
            self.Y[level] =  np.sqrt(self.sample_size) * np.transpose(Q)
            vectors = []
            for i in range(self.R[level]):
                vectors.append(self.U[level][i].vector()[:])
            vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
            for i in range(self.R[level]):
                self.U[level][i].vector()[:] = vectors[i]


    #explicit scheme
    def explicit_simulate(self,end = 2.5):
        t = 0
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        v = []
        for level in range(self.level):
            for i in range(self.R[level]):
                self.U[level][i] = TrialFunction(self.V[level])
            v.append(TestFunction(self.V[level]))  
        
        lhs = []
        rhs = []
        for level in range(self.level):
            lhs_component = []
            rhs_component = []
            for i in range(self.R[level]):
                a_i = self.U[level][i] * v[level] * dx
                L_i = self.U_n[level][i] * v[level] * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y[level][i],level), grad(v[level])) * dx
                lhs_component.append(a_i)
                rhs_component.append(L_i)
                self.U[level][i] = Function(self.V[level])
            lhs.append(lhs_component)
            rhs.append(rhs_component)
        
        
        
        while t < end:           
            # Compute solution
            for level in range(self.level):
                for i in range(self.R[level]):
                    solve(lhs[level][i]==rhs[level][i],self.U[level][i],self.bc[level])

                self.matrix_calculate(level)

                A = []
                for i in range(self.R[level]):
                    A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a,level)[i],level))
                A = np.array(A)
                det = np.linalg.det(self.matrix[level])
                if np.isclose(det, 0):
                    self.Y[level] += -self.dt * scipy.linalg.lstsq(self.matrix[level],A)[0]
                else:
                    self.Y[level] += -self.dt * scipy.linalg.solve(self.matrix[level],A)
                
                
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            
            # assign
            for level in range(self.level):
                for j in range(self.R[level]):
                    self.U_n[level][j].assign(self.U[level][j])
            self.Y_n = self.Y
            

            self.timelist.append(t)
            energy = self.energynorm()
            self.energylist.append(energy)
            l2 = self.exl2norm()
            self.L2list.append(l2)
            print("time: ",t)
            print("energy norm: ", energy )
            print("L2 norm: ", l2 )
            if energy > 10 ** 7:
                break 
                    
        self.plot_field()
        self.plot_norm()

        

    # Energy norm
    def energynorm(self):
        u = Function(self.V[self.level -1])
        for level in range(self.level):
            for j in range(self.R[level]):
                u.vector()[:] += interpolate(self.U[level][j],self.V[self.level -1]).vector()[:] * self.Y[level][j][0]  
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        
        for i in range(1, self.sample_size):
            u = Function(self.V[self.level -1])
            for level in range(self.level):
                for j in range(self.R[level]):
                    u.vector()[:] += interpolate(self.U[level][j],self.V[self.level -1]).vector()[:] * self.Y[level][j][i]  
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy

    def exl2norm(self):
        u = Function(self.V[self.level -1])
        for level in range(self.level):  
            for j in range(self.R[level]):
                u.vector()[:] += interpolate(self.U[level][j],self.V[self.level -1]).vector()[:] * self.Y[level][j][0]
        form = u * u * dx
        
        for i in range(1, self.sample_size):
            u = Function(self.V[self.level -1])
            for level in range(self.level):
                for j in range(self.R[level]):
                    u.vector()[:] += interpolate(self.U[level][j],self.V[self.level -1]).vector()[:] * self.Y[level][j][i] 
            form += u * u * dx
        exl2 = assemble(form) / self.sample_size
        return np.sqrt(exl2)

            


    
    # monitor energy norm
    def plot_norm(self):
            plt.plot(self.timelist, self.energylist)
            plt.plot(self.timelist, self.L2list)
            plt.yscale('log')
            plt.xlabel("time")
            plt.ylabel("norm")
            plt.legend(['energy norm','L2 norm'])
            plt.show()
        
    #visualize mean function of the random field
    def plot_field(self):
        u = Function(self.V[self.level -1])
        for level in range(self.level):
            for i in range(self.R[level]):
                 u.vector()[:] += interpolate(self.U[level][i],self.V[self.level -1]).vector()[:] * np.mean(self.Y[level][i])
        plot(u)
        plt.show()
        


    
