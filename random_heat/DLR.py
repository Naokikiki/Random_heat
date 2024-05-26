#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from fenics import *
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


"""
Example
2D random heat equation with forcing term f = 0
 
 
  du/dt = - div(a(ξ)∇u)   in the unit square
  u = 0                   on the boundary
  u = u_0                 at t = 0
 
  D = [0,1]^2  is physical space
  Τ = [-1,1]^M  is sample space

a(x,ξ) = 0.3 + sum_{m=1}^M (cos(2πmx1)+cos(2πmx2)/m^2π^2 * ξm)
u_0 = 10 sin(πx1) sin(πx2) + 4/3 sin(6πx1) sin(6πx2) + 2 sin(2πx1) sin(2πx2)ξ1 + 2 sin(4πx1) sin(4πx2)ξ2 + 2 sin(6πx1) sin(6πx2)(ξ1^2 − E[ξ1^2])

"""

class DLR:
    def __init__(self,dt,n,M=10,R = 3,sample_size = 50,mesh_type='2D',mean =None,U=None,Y=None,a=None,a_0=None,a_sto=None):
        self.dt = dt  # time step
        self.n = n  # number of mesh
        self.h = 1 / n  # element size
        self.M = M  # The number of random variables( The number of truncation of a(ξ))
        self.R = R  # rank of DLR
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh = UnitIntervalMesh(self.n)
        elif mesh_type == '2D':
            self.mesh = UnitSquareMesh(self.n, self.n)
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V = FunctionSpace(self.mesh, 'P', 1)
        # self.V_R = MixedFunctionSpace([self.V] * self.R)

        ## Define boundary condition
        self.bc = DirichletBC(self.V, Constant(0), 'on_boundary')

        
        ## Initialize functions 
        #set random values
        self.sampled = self.sampling() #(M,self.sample_size) ,unifrom random values in T
        ksi_1 = self.sampled[0]
        ksi_2 = self.sampled[1]

        # E[u]
        if mean == None:
            mean = Expression('10 * sin(pi * x[0]) * sin(pi * x[1]) + 4/3 * sin(6 * pi * x[0]) * sin(6 * pi * x[1])',degree=3)
        self.mean = interpolate(mean,self.V)  #(V_h)
        self.mean_n = interpolate(mean,self.V) 
        
         # stochatic basis functions   
        if Y == None:
            self.Y = [] #(R,sample_size) # R=3
            Y_1 = ksi_1
            self.Y.append(Y_1)
            Y_2 = ksi_2
            self.Y.append(Y_2)
            Y_3 = ksi_1 ** 2 - 2/3
            self.Y.append(Y_3)
        else:
            self.Y = Y #(R,sample_size)
        self.Y = np.array(self.Y)
        
        # deterministic basis functions
        if U == None:
            U = [Expression('2 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])',degree=3),Expression('2 * sin(4 * pi * x[0]) * sin(4 * pi * x[1])',degree=3),Expression('2 * sin(6 * pi * x[0]) * sin(6 * pi * x[1])',degree=3)]
        self.U = []#  (R,V_h) 
        self.U_n = []
        for u_i in U:
            self.U.append(interpolate(u_i,self.V))
            self.U_n.append(interpolate(u_i,self.V))
        


        ## set coefficient
        if a == None:
            self.a_0 = Constant(0.3) #(L^♾️)
            self.a_sto = [] #(sample_size,L^♾️)
            self.a = [] #(sample_size,L^♾️)
            for i in range(self.sample_size):
                str_expr = ""
                for j in range(1, self.M + 1):
                    str_expr += f"( cos(2 * pi * {j} * x[0] ) + cos(2 * pi * {j} * x[1] ) ) / ({j} * {j} * pi * pi) * sampled[{j-1}] + "
                str_expr = str_expr[:-3]
                self.a_sto.append(Expression(str_expr, degree=3, sampled=Constant(self.sampled[:,i])))
                str_expr += " + a_0"
                self.a.append(Expression(str_expr, degree=3, sampled=Constant(self.sampled[:,i]),a_0 = Constant(0.3)))
            # for i in range(self.sample_size):
            #     self.a[i] = interpolate(self.a[i],self.V)
        else:
            self.a_0 = a_0
            self.a_sto = a_sto
            self.a = a

        #M_i,j = <U_i,U_j>
        self.matrix = np.zeros((R,R)) 

        # Mass matrix
        tri = TrialFunction(self.V)
        test = TestFunction(self.V)
        integral = tri * test * dx
        A = assemble(integral)
        # Convert the matrix to a NumPy array for easier inspection
        self.mass_matrix = as_backend_type(A).mat().getValues(range(A.size(0)), range(A.size(1)))


        # print("energy",self.energynorm())
        # print("l2",self.exl2norm())
        
        

        for i in range(self.R):
            Y_i_mean = np.mean(self.Y[i])
            self.mean.vector()[:] += self.U[i].vector()[:] * Y_i_mean
            self.Y[i] -= Y_i_mean   
        self.mean_n.assign(self.mean) 
        
        self.reorthogonalize()
         
        for i in range(self.R):
            self.U_n[i].assign(self.U[i])
        # print("energy",self.energynorm())
        # print("l2",self.exl2norm())
        

         
                   
        ##for plot
        self.timelist = []
        self.energylist = []
        self.L2list= []

    
    def sampling(self):
        return np.random.uniform(low=-1.0, high=1.0, size=(self.M,self.sample_size))

    # calculate quadratures, M_i,j = <U_i,U_j>
    def matrix_calculate(self):
        for i in range(self.R):
            for j in range(i,self.R):
                value = assemble((self.U[i]*self.U[j]) * dx)
                self.matrix[i][j] = value
                self.matrix[j][i] = value

    ## subfunctions for calculating dinamics

    def E_a_grad_u(self,a):
        grad_list = []
        for i in range(self.sample_size):
            func = self.mean_n
            for j in range(self.R):
                func += (self.U_n[j] * self.Y[j][i])
            grad_list.append(grad(func))
        ans = a[0]* grad_list[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i])
        ans /= self.sample_size
        return ans
    

    def E_a_grad_u_Y(self,a,Y):
        grad_list = []
        for i in range(self.sample_size):
            func = self.mean_n
            for j in range(self.R):
                func += (self.U_n[j] * self.Y[j][i])
            grad_list.append(grad(func))
        ans = a[0]* grad_list[0]* Y[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i]* Y[i])
        ans /= self.sample_size
        return ans
    
    

    

    def a_grad_u_grad_U(self,a):
        ans = np.zeros((self.R,self.sample_size)) 
        u = []
        for i in range(self.sample_size):
            func = self.mean
            for j in range(self.R):
                func += (self.U_n[j] * Constant(self.Y[j][i]) )
            u.append(func)
        for i in range(self.R):
            for j in range(self.sample_size):
                ans[i][j] = assemble(a[j] * dot(grad(u[j]),grad(self.U[i])) * dx)
        row_means = np.mean(ans, axis=1)
        ans -= row_means[:, np.newaxis]

        return ans
    
    def dt_a0_grad_U_grad_U(self):
        ans = np.zeros((self.R,self.R)) 
        for i in range(self.R):
            for j in range(i,self.R):
                value = self.dt *  assemble(self.a_0 * dot(grad(self.U[i]),grad(self.U[j])) * dx)
                ans[i][j] = value
                ans[j][i] = value
        return ans
    


    # def pre_assemble(self):
    #     self.u_1 = Function(self.V)
    #     self.u_2 = Function(self.V)
    #     self.u_3 = Function(self.V)
    #     self.agradugradv = self.u_3 * dot(grad(self.u_1),grad(self.u_2)) * dx
    #     assemble(self.a)

    # def a_grad_u_grad_U(self,a):
    #     ans = np.zeros((self.R,self.sample_size)) 
    #     u = []
    #     for i in range(self.sample_size):
    #         u.append(self.mean + self.U[0] * Constant(self.Y[0][i]) + self.U[1] * Constant(self.Y[1][i]) + self.U[2] * Constant(self.Y[2][i]))
    #     for i in range(self.R):
    #         for j in range(self.sample_size):
    #             ans[i][j] = assemble(self.agradugradv(coefficients={self.u_3=a[j],self.u_1 = u[j],self.u_2 = self.U[i]}))
    #                 # self.a[j] * dot(grad(u[j]),grad(self.U[i])) * dx)
    #     return ans
                             
                          

    def orthogonal_projection(self,v):
        ortho = np.inner(v,self.Y[0]) / self.sample_size * self.Y[0]
        for i in range(1,self.R):
            ortho += np.inner(v,self.Y[i]) / self.sample_size * self.Y[i]
        return v - ortho
    
    # def reorthogonalize(self):
    #     return

    def reorthogonalize(self):
        U_vectors = []
        v2d = vertex_to_dof_map(self.V)
        for i in range(self.R):
            U_vectors.append(self.U[i].vector()[v2d])
        U_vectors = np.array(U_vectors)
        UY = U_vectors.T @ self.Y
        Matrix = UY.T @ self.mass_matrix @ UY
        U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
        Vt_reduced = Vt[:self.R, :]

        self.Y = Vt_reduced 
        U_vectors = self.Y @ UY.T
        self. Y *= np.sqrt(self.sample_size)
        for i in range(self.R):
            self.U[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
        

    
    # def reorthogonalize(self): #dimension :Y(self.R,self.sample_size), U(self.R,V)
    #     U_vectors = []
    #     for i in range(self.R):
    #         U_vectors.append(self.U[i].vector()[:])
    #     U, S, Vt = np.linalg.svd(np.matmul(np.transpose(U_vectors),self.Y), full_matrices=False)    
    #     U_reduced = U[:, :self.R]      # N x R
    #     S_reduced = np.diag(S[:self.R])  # R x R
    #     Vt_reduced = Vt[:self.R, :]    # R x M
        
    #     U_vectors =np.transpose(np.matmul(U_reduced,S_reduced) / np.sqrt(self.sample_size))
    #     for i in range(self.R):
    #         self.U[i].vector()[:] = U_vectors[i]
    #     self.Y = Vt_reduced * np.sqrt(self.sample_size)

    # def reorthogonalize(self): #dimension :Y(self.R,self.sample_size), U(self.R,V)
    #     Q, _ = np.linalg.qr(np.transpose(self.Y))
    #     # self.Y = np.transpose(Q) 
    #     self.Y = np.transpose(Q) *np.sqrt(self.sample_size)
    #     vectors = []
    #     for i in range(self.R):
    #         vectors.append(self.U[i].vector()[:])
    #     vectors = np.matmul(_,vectors)
    #     for i in range(self.R):
    #         # self.U[i].vector()[:] = vectors[i] 
    #         self.U[i].vector()[:] = vectors[i] /np.sqrt(self.sample_size)




    #explicit scheme
    def explicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        self.mean = TrialFunction(self.V)
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        a_mean = self.mean * v * dx  
        L_mean = self.mean_n * v * dx - self.dt * dot(self.E_a_grad_u(self.a), grad(v)) * dx
        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
           
        self.mean = Function(self.V)
        for i in range(self.R):
            self.U[i] = Function(self.V)
        
        while t < end:           
            # Compute solution
            solve(a_mean == L_mean, self.mean, self.bc) 
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()

           
            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a)[i]))
            A = np.array(A)
            det = np.linalg.det(self.matrix)
            if np.isclose(det, 0):
                self.Y += -self.dt * scipy.linalg.lstsq(self.matrix,A)[0]
            else:
                self.Y += -self.dt * scipy.linalg.solve(self.matrix,A)
            #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
            
            self.mean_n.assign(self.mean) 
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
           
            if count % 1 == 0:
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

        

    #semi implicit scheme
    def semi_implicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        self.mean = TrialFunction(self.V)
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        a_mean = self.mean * v * dx  + self.dt * self.a_0 * dot(grad(self.mean),grad(v)) * dx
        L_mean = self.mean_n * v * dx - self.dt * dot(self.E_a_grad_u(self.a_sto),grad(v)) * dx
        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx + self.dt * self.a_0 * dot(grad(self.U[i]),grad(v)) * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a_sto,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
   
        self.mean = Function(self.V)
        for i in range(self.R):
            self.U[i] =  Function(self.V)

        while t < end:           
            # Compute solution
            solve(a_mean == L_mean, self.mean, self.bc) 
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()
            
            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a_sto)[i]))
            A = np.array(A)
            self.Y += -self.dt * scipy.linalg.solve(self.matrix+ self.dt_a0_grad_U_grad_U(),A)
            
            #             self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix + self.dt_a0_grad_U_grad_U()),A)
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
            
            self.mean_n.assign(self.mean) 
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
            
            if count % 1 == 0:
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

        
        

   #energy norm

    def energynorm(self):
        u = self.mean 
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        for i in range(1,self.sample_size):
            u = self.mean
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy
    
    def exl2norm(self):
        u = self.mean
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = u* u * dx
        for i in range(1,self.sample_size):
            u = self.mean
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += u * u * dx
        exl2 = assemble(form) / self.sample_size
        return np.sqrt(exl2)
    
    # def exl2norm(self):
    #     return norm(self.mean,'l2')
    
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
        # u = self.mean + self.U[0] * np.mean(self.Y[0]) + self.U[1] * np.mean(self.Y[1]) + self.U[2] * np.mean(self.Y[2])
        plot(self.mean)
        plt.show()
        


    

   
    


# Not separate mean

# In[85]:


"""
Example
2D random heat equation with forcing term f = 0
 
 
  du/dt = - div(a(ξ)∇u)   in the unit square
  u = 0                   on the boundary
  u = u_0                 at t = 0
 
  D = [0,1]^2  is physical space
  Τ = [-1,1]^M  is sample space

a(x,ξ) = 0.3 + sum_{m=1}^M (cos(2πmx1)+cos(2πmx2)/m^2π^2 * ξm)
u_0 = 10 sin(πx1) sin(πx2) + 4/3 sin(6πx1) sin(6πx2) + 2 sin(2πx1) sin(2πx2)ξ1 + 2 sin(4πx1) sin(4πx2)ξ2 + 2 sin(6πx1) sin(6πx2)(ξ1^2 − E[ξ1^2])

"""

class DLR2:
    def __init__(self,dt,n,U,Y,a,a_0,a_sto,M=10,R = 3,sample_size = 50,mesh_type='2D'):
        self.dt = dt  # time step
        self.n = n  # number of mesh
        self.h = 1 / n  # element size
        self.M = M  # The number of random variables( The number of truncation of a(ξ))
        self.R = R  # rank of DLR
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh = UnitIntervalMesh(self.n)
        elif mesh_type == '2D':
            self.mesh = UnitSquareMesh(self.n, self.n)
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V = FunctionSpace(self.mesh, 'P', 1)
        # self.V_R = MixedFunctionSpace([self.V] * self.R)

        ## Define boundary condition
        self.bc = DirichletBC(self.V, Constant(0), 'on_boundary')

        
        ## Initialize functions 
       
         # stochatic basis functions        
        self.Y = Y #(R,sample_size)
        self.Y = np.array(self.Y)
        
        # deterministic basis functions
        self.U = []#  (R,V_h) 
        self.U_n = []
        for u_i in U:
            self.U.append(interpolate(u_i,self.V))
            self.U_n.append(interpolate(u_i,self.V))

        ## set coefficient
        self.a_0 = a_0
        self.a_sto = a_sto
        self.a = a 
        

        #M_i,j = <U_i,U_j>
        self.matrix = np.zeros((R,R)) 

        # Mass matrix
        tri = TrialFunction(self.V)
        test = TestFunction(self.V)
        integral = tri * test * dx
        A = assemble(integral)
        # Convert the matrix to a NumPy array for easier inspection
        self.mass_matrix = as_backend_type(A).mat().getValues(range(A.size(0)), range(A.size(1)))


        # print("hi",self.energynorm())
        # print("hi",self.exl2norm())
        # print(np.mean(self.Y[0]))
        # print(np.mean(self.Y[1]))
        # print(np.mean(self.Y[2]))
       
        self.reorthogonalize()
        # print("hi",self.energynorm())
        # print("hi",self.exl2norm())
        # print(np.matmul(self.Y,np.transpose(self.Y)))
        # print(np.mean(self.Y[0]))
        # print(np.mean(self.Y[1]))
        # print(np.mean(self.Y[2]))
        
        for i in range(self.R):
            self.U_n[i].assign(self.U[i])
        

        ##for plot
        self.timelist = []
        self.energylist = []
        self.L2list= []

   
    # calculate quadratures, M_i,j = <U_i,U_j>
    def matrix_calculate(self):
        for i in range(self.R):
            for j in range(i,self.R):
                value = assemble((self.U[i]*self.U[j]) * dx)
                self.matrix[i][j] = value
                self.matrix[j][i] = value

    ## subfunctions for calculating dinamics
    

    def E_a_grad_u_Y(self,a,Y):
        grad_list = []
        for i in range(self.sample_size):
            func = Constant(0)
            for j in range(self.R):
                func += (self.U_n[j] * self.Y[j][i])
            grad_list.append(grad(func))
        ans = a[0]* grad_list[0]* Y[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i]* Y[i])
        ans /= self.sample_size
        return ans
    
    

    

    def a_grad_u_grad_U(self,a):
        ans = np.zeros((self.R,self.sample_size)) 
        u = []
        for i in range(self.sample_size):
            func = Constant(0)
            for j in range(self.R):
                func += (self.U_n[j] * Constant(self.Y[j][i]) )
            u.append(func)
        for i in range(self.R):
            for j in range(self.sample_size):
                ans[i][j] = assemble(dot(a[j] *grad(u[j]),grad(self.U[i])) * dx)
        
        return ans
    
    def dt_a0_grad_U_grad_U(self):
        ans = np.zeros((self.R,self.R)) 
        for i in range(self.R):
            for j in range(i,self.R):
                value = self.dt *  assemble(self.a_0 * dot(grad(self.U[i]),grad(self.U[j])) * dx)
                ans[i][j] = value
                ans[j][i] = value
        return ans
    


    # def pre_assemble(self):
    #     self.u_1 = Function(self.V)
    #     self.u_2 = Function(self.V)
    #     self.u_3 = Function(self.V)
    #     self.agradugradv = self.u_3 * dot(grad(self.u_1),grad(self.u_2)) * dx
    #     assemble(self.a)

    # def a_grad_u_grad_U(self,a):
    #     ans = np.zeros((self.R,self.sample_size)) 
    #     u = []
    #     for i in range(self.sample_size):
    #         u.append(self.mean + self.U[0] * Constant(self.Y[0][i]) + self.U[1] * Constant(self.Y[1][i]) + self.U[2] * Constant(self.Y[2][i]))
    #     for i in range(self.R):
    #         for j in range(self.sample_size):
    #             ans[i][j] = assemble(self.agradugradv(coefficients={self.u_3=a[j],self.u_1 = u[j],self.u_2 = self.U[i]}))
    #                 # self.a[j] * dot(grad(u[j]),grad(self.U[i])) * dx)
    #     return ans
                             
                          

    def orthogonal_projection(self,v):
        ortho = np.inner(v,self.Y[0]) / self.sample_size * self.Y[0]
        for i in range(1,self.R):
            ortho += np.inner(v,self.Y[i]) / self.sample_size * self.Y[i]
        return v - ortho
    
    # def reorthogonalize(self):
    #     return

    # def reorthogonalize(self):
    #     U_vectors = []
    #     v2d = vertex_to_dof_map(self.V)
    #     for i in range(self.R):
    #         U_vectors.append(self.U[i].vector()[v2d])
    #     U_vectors = np.array(U_vectors)
    #     UY = U_vectors.T @ self.Y
    #     Matrix = UY.T @ self.mass_matrix @ UY
    #     U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
    #     Vt_reduced = Vt[:self.R, :]

    #     self.Y = Vt_reduced 
    #     U_vectors = self.Y @ UY.T
    #     self. Y *= np.sqrt(self.sample_size)
    #     for i in range(self.R):
    #         self.U[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
            
    # def reorthogonalize(self): #dimension :Y(self.R,self.sample_size), U(self.R,V)
    #     U_vectors = []
    #     for i in range(self.R):
    #         U_vectors.append(self.U[i].vector()[:])
    #     U, S, Vt = np.linalg.svd(np.matmul(np.transpose(U_vectors),self.Y), full_matrices=False)    
    #     U_reduced = U[:, :self.R]      # N x R
    #     S_reduced = np.diag(S[:self.R])  # R x R
    #     Vt_reduced = Vt[:self.R, :]    # R x M
        
    #     U_vectors =np.transpose(np.matmul(U_reduced,S_reduced) / np.sqrt(self.sample_size))
    #     for i in range(self.R):
    #         self.U[i].vector()[:] = U_vectors[i]
    #     self.Y = Vt_reduced * np.sqrt(self.sample_size)


    def reorthogonalize(self):
        Q, _ = np.linalg.qr(np.transpose(self.Y))
        # self.Y =  np.transpose(Q)
        self.Y =  np.sqrt(self.sample_size) * np.transpose(Q)
        vectors = []
        for i in range(self.R):
            vectors.append(self.U[i].vector()[:])
        # vectors = np.matmul(_ ,vectors)
        vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
        for i in range(self.R):
            self.U[i].vector()[:] = vectors[i]


    #explicit scheme
    def explicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
           
        for i in range(self.R):
            self.U[i] = Function(self.V)
        
        while t < end:           
            # Compute solution
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()

            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a)[i]))
            A = np.array(A)
            det = np.linalg.det(self.matrix)
            if np.isclose(det, 0):
                self.Y += -self.dt * scipy.linalg.lstsq(self.matrix,A)[0]
            else:
                self.Y += -self.dt * scipy.linalg.solve(self.matrix,A)
                #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
                
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
            
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
           
            if count % 1 == 0:
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

        

    #semi implicit scheme
    def semi_implicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        
        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx + self.dt * self.a_0 * dot(grad(self.U[i]),grad(v)) * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a_sto,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
   
        for i in range(self.R):
            self.U[i] =  Function(self.V)

        while t < end:           
            # Compute solution
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()
            
            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a_sto)[i]))
            A = np.array(A)
            self.Y += -self.dt * scipy.linalg.solve(self.matrix+ self.dt_a0_grad_U_grad_U(),A)
            
            #             self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix + self.dt_a0_grad_U_grad_U()),A)
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
             
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
            
            if count % 1 == 0:
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

        
        

   #energy norm

    def energynorm(self):
        u = Constant(0)
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        for i in range(1,self.sample_size):
            u = Constant(0)
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy
    
    def exl2norm(self):
        u = Constant(0)
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = u* u * dx
        for i in range(1,self.sample_size):
            u = Constant(0)
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += u* u * dx
        exl2 = assemble(form) / self.sample_size
        return exl2
            


    
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
        u = 0
        for i in range(self.R):
            u += self.U[i] * np.mean(self.Y[i])
        plot(u)
        plt.show()
        


    

   
"""
Example
2D random heat equation with forcing term f = 0
 
 
  du/dt = - div(a(ξ)∇u)   in the unit square
  u = 0                   on the boundary
  u = u_0                 at t = 0
 
  D = [0,1]^2  is physical space
  Τ = [-1,1]^M  is sample space

a(x,ξ) = 0.3 + sum_{m=1}^M (cos(2πmx1)+cos(2πmx2)/m^2π^2 * ξm)
u_0 = 10 sin(πx1) sin(πx2) + 4/3 sin(6πx1) sin(6πx2) + 2 sin(2πx1) sin(2πx2)ξ1 + 2 sin(4πx1) sin(4πx2)ξ2 + 2 sin(6πx1) sin(6πx2)(ξ1^2 − E[ξ1^2])

"""

class DLR3:
    def __init__(self,dt,n,M=10,R = 3,sample_size = 50,mesh_type='2D',mean =None,U=None,Y=None,a=None,a_0=None,a_sto=None):
        self.dt = dt  # time step
        self.n = n  # number of mesh
        self.h = 1 / n  # element size
        self.M = M  # The number of random variables( The number of truncation of a(ξ))
        self.R = R  # rank of DLR
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh = UnitIntervalMesh(self.n)
        elif mesh_type == '2D':
            self.mesh = UnitSquareMesh(self.n, self.n)
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V = FunctionSpace(self.mesh, 'P', 1)
        # self.V_R = MixedFunctionSpace([self.V] * self.R)

        ## Define boundary condition
        self.bc = DirichletBC(self.V, Constant(0), 'on_boundary')

        
        ## Initialize functions 
        #set random values
        self.sampled = self.sampling() #(M,self.sample_size) ,unifrom random values in T
        ksi_1 = self.sampled[0]
        ksi_2 = self.sampled[1]

        # E[u]
        if mean == None:
            mean = Expression('10 * sin(pi * x[0]) * sin(pi * x[1]) + 4/3 * sin(6 * pi * x[0]) * sin(6 * pi * x[1])',degree=3)
        self.mean = interpolate(mean,self.V)  #(V_h)
        self.mean_n = interpolate(mean,self.V) 
        
         # stochatic basis functions   
        if Y == None:
            self.Y = [] #(R,sample_size) # R=3
            Y_1 = ksi_1
            self.Y.append(Y_1)
            Y_2 = ksi_2
            self.Y.append(Y_2)
            Y_3 = ksi_1 ** 2 - 2/3
            self.Y.append(Y_3)
        else:
            self.Y = Y #(R,sample_size)
        self.Y = np.array(self.Y)
        
        # deterministic basis functions
        if U == None:
            U = [Expression('2 * sin(2 * pi * x[0]) * sin(2 * pi * x[1])',degree=3),Expression('2 * sin(4 * pi * x[0]) * sin(4 * pi * x[1])',degree=3),Expression('2 * sin(6 * pi * x[0]) * sin(6 * pi * x[1])',degree=3)]
        self.U = []#  (R,V_h) 
        self.U_n = []
        for u_i in U:
            self.U.append(interpolate(u_i,self.V))
            self.U_n.append(interpolate(u_i,self.V))
        


        ## set coefficient
        if a == None:
            self.a_0 = Constant(0.3) #(L^♾️)
            self.a_sto = [] #(sample_size,L^♾️)
            self.a = [] #(sample_size,L^♾️)
            for i in range(self.sample_size):
                str_expr = ""
                for j in range(1, self.M + 1):
                    str_expr += f"( cos(2 * pi * {j} * x[0] ) + cos(2 * pi * {j} * x[1] ) ) / ({j} * {j} * pi * pi) * sampled[{j-1}] + "
                str_expr = str_expr[:-3]
                self.a_sto.append(Expression(str_expr, degree=3, sampled=Constant(self.sampled[:,i])))
                str_expr += " + a_0"
                self.a.append(Expression(str_expr, degree=3, sampled=Constant(self.sampled[:,i]),a_0 = Constant(0.3)))
            # for i in range(self.sample_size):
            #     self.a[i] = interpolate(self.a[i],self.V)
        else:
            self.a_0 = a_0
            self.a_sto = a_sto
            self.a = a

        #M_i,j = <U_i,U_j>
        self.matrix = np.zeros((R,R)) 

        # Mass matrix
        tri = TrialFunction(self.V)
        test = TestFunction(self.V)
        integral = tri * test * dx
        A = assemble(integral)
        # Convert the matrix to a NumPy array for easier inspection
        self.mass_matrix = as_backend_type(A).mat().getValues(range(A.size(0)), range(A.size(1)))


        # print("energy",self.energynorm())
        # print("l2",self.exl2norm())
        
        

        for i in range(self.R):
            Y_i_mean = np.mean(self.Y[i])
            self.mean.vector()[:] += self.U[i].vector()[:] * Y_i_mean
            self.Y[i] -= Y_i_mean   
        self.mean_n.assign(self.mean) 
        
        self.reorthogonalize()
 
        for i in range(self.R):
            self.U_n[i].assign(self.U[i])
        # print("energy",self.energynorm())
        # print("l2",self.exl2norm())
        

         
                   
        ##for plot
        self.timelist = []
        self.energylist = []
        self.L2list= []

    
    def sampling(self):
        return np.random.uniform(low=-1.0, high=1.0, size=(self.M,self.sample_size))

    # calculate quadratures, M_i,j = <U_i,U_j>
    def matrix_calculate(self):
        for i in range(self.R):
            for j in range(i,self.R):
                value = assemble((self.U[i]*self.U[j]) * dx)
                self.matrix[i][j] = value
                self.matrix[j][i] = value

    ## subfunctions for calculating dinamics

    def E_a_grad_u(self,a):
        grad_list = []
        for i in range(self.sample_size):
            func = self.mean_n
            for j in range(self.R):
                func += (self.U_n[j] * self.Y[j][i])
            grad_list.append(grad(func))
        ans = a[0]* grad_list[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i])
        ans /= self.sample_size
        return ans
    

    def E_a_grad_u_Y(self,a,Y):
        grad_list = []
        for i in range(self.sample_size):
            func = self.mean_n
            for j in range(self.R):
                func += (self.U_n[j] * self.Y[j][i])
            grad_list.append(grad(func))
        ans = a[0]* grad_list[0]* Y[0]
        for i in range(1,self.sample_size):
            ans += (a[i]* grad_list[i]* Y[i])
        ans /= self.sample_size
        return ans
    
    

    

    def a_grad_u_grad_U(self,a):
        ans = np.zeros((self.R,self.sample_size)) 
        u = []
        for i in range(self.sample_size):
            func = self.mean
            for j in range(self.R):
                func += (self.U_n[j] * Constant(self.Y[j][i]) )
            u.append(func)
        for i in range(self.R):
            for j in range(self.sample_size):
                ans[i][j] = assemble(a[j] * dot(grad(u[j]),grad(self.U[i])) * dx)
        row_means = np.mean(ans, axis=1)
        ans -= row_means[:, np.newaxis]

        return ans
    
    def dt_a0_grad_U_grad_U(self):
        ans = np.zeros((self.R,self.R)) 
        for i in range(self.R):
            for j in range(i,self.R):
                value = self.dt *  assemble(self.a_0 * dot(grad(self.U[i]),grad(self.U[j])) * dx)
                ans[i][j] = value
                ans[j][i] = value
        return ans
    


    # def pre_assemble(self):
    #     self.u_1 = Function(self.V)
    #     self.u_2 = Function(self.V)
    #     self.u_3 = Function(self.V)
    #     self.agradugradv = self.u_3 * dot(grad(self.u_1),grad(self.u_2)) * dx
    #     assemble(self.a)

    # def a_grad_u_grad_U(self,a):
    #     ans = np.zeros((self.R,self.sample_size)) 
    #     u = []
    #     for i in range(self.sample_size):
    #         u.append(self.mean + self.U[0] * Constant(self.Y[0][i]) + self.U[1] * Constant(self.Y[1][i]) + self.U[2] * Constant(self.Y[2][i]))
    #     for i in range(self.R):
    #         for j in range(self.sample_size):
    #             ans[i][j] = assemble(self.agradugradv(coefficients={self.u_3=a[j],self.u_1 = u[j],self.u_2 = self.U[i]}))
    #                 # self.a[j] * dot(grad(u[j]),grad(self.U[i])) * dx)
    #     return ans
                             
                          

    def orthogonal_projection(self,v):
        ortho = np.inner(v,self.Y[0]) / self.sample_size * self.Y[0]
        for i in range(1,self.R):
            ortho += np.inner(v,self.Y[i]) / self.sample_size * self.Y[i]
        return v - ortho
    
    # def reorthogonalize(self):
    #     return

    # def reorthogonalize(self):
    #     U_vectors = []
    #     v2d = vertex_to_dof_map(self.V)
    #     for i in range(self.R):
    #         U_vectors.append(self.U[i].vector()[v2d])
    #     U_vectors = np.array(U_vectors)
    #     UY = U_vectors.T @ self.Y
    #     Matrix = UY.T @ self.mass_matrix @ UY
    #     U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
    #     Vt_reduced = Vt[:self.R, :]

    #     self.Y = Vt_reduced 
    #     U_vectors = self.Y @ UY.T
    #     self. Y *= np.sqrt(self.sample_size)
    #     for i in range(self.R):
    #         self.U[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
        

    
    # def reorthogonalize(self): #dimension :Y(self.R,self.sample_size), U(self.R,V)
    #     U_vectors = []
    #     for i in range(self.R):
    #         U_vectors.append(self.U[i].vector()[:])
    #     U, S, Vt = np.linalg.svd(np.matmul(np.transpose(U_vectors),self.Y), full_matrices=False)    
    #     U_reduced = U[:, :self.R]      # N x R
    #     S_reduced = np.diag(S[:self.R])  # R x R
    #     Vt_reduced = Vt[:self.R, :]    # R x M
        
    #     U_vectors =np.transpose(np.matmul(U_reduced,S_reduced) / np.sqrt(self.sample_size))
    #     for i in range(self.R):
    #         self.U[i].vector()[:] = U_vectors[i]
    #     self.Y = Vt_reduced * np.sqrt(self.sample_size)

    def reorthogonalize(self): #dimension :Y(self.R,self.sample_size), U(self.R,V)
        Q, _ = np.linalg.qr(np.transpose(self.Y))
        # self.Y = np.transpose(Q) 
        self.Y = np.transpose(Q) *np.sqrt(self.sample_size)
        vectors = []
        for i in range(self.R):
            vectors.append(self.U[i].vector()[:])
        vectors = np.matmul(_,vectors)
        for i in range(self.R):
            # self.U[i].vector()[:] = vectors[i] 
            self.U[i].vector()[:] = vectors[i] /np.sqrt(self.sample_size)




    #explicit scheme
    def explicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        self.mean = TrialFunction(self.V)
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        a_mean = self.mean * v * dx  
        L_mean = self.mean_n * v * dx - self.dt * dot(self.E_a_grad_u(self.a), grad(v)) * dx
        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
           
        self.mean = Function(self.V)
        for i in range(self.R):
            self.U[i] = Function(self.V)
        
        while t < end:           
            # Compute solution
            solve(a_mean == L_mean, self.mean, self.bc) 
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()

            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a)[i]))
            A = np.array(A)
            det = np.linalg.det(self.matrix)
            if np.isclose(det, 0):
                self.Y += -self.dt * scipy.linalg.lstsq(self.matrix,A)[0]
            else:
                self.Y += -self.dt * scipy.linalg.solve(self.matrix,A)
            #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
            
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
            
            self.mean_n.assign(self.mean) 
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
           
            if count % 1 == 0:
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

        

    #semi implicit scheme
    def semi_implicit_simulate(self,end = 2.5):
        t = 0
        count = 0 # calculate energynorm each step is cotly so only every some steps
        self.timelist.append(t)
        energy = self.energynorm()
        self.energylist.append(energy)
        l2 = self.exl2norm()
        self.L2list.append(l2)
        print("time: ",t)
        print("energy norm: ", energy )
        print("L2 norm: ", l2 )

        # Define variational problem
        
        self.mean = TrialFunction(self.V)
        for i in range(self.R):
            self.U[i] = TrialFunction(self.V)
        v = TestFunction(self.V)

        a_mean = self.mean * v * dx  + self.dt * self.a_0 * dot(grad(self.mean),grad(v)) * dx
        L_mean = self.mean_n * v * dx - self.dt * dot(self.E_a_grad_u(self.a_sto),grad(v)) * dx
        lhs = []
        rhs = []
        for i in range(self.R):
            a_i = self.U[i] * v * dx + self.dt * self.a_0 * dot(grad(self.U[i]),grad(v)) * dx
            L_i = self.U_n[i] * v * dx - self.dt * dot(self.E_a_grad_u_Y(self.a_sto,self.Y[i]), grad(v)) * dx
            lhs.append(a_i)
            rhs.append(L_i)
   
        self.mean = Function(self.V)
        for i in range(self.R):
            self.U[i] =  Function(self.V)

        while t < end:           
            # Compute solution
            solve(a_mean == L_mean, self.mean, self.bc) 
            for i in range(self.R):
                solve(lhs[i]==rhs[i],self.U[i],self.bc)
         
            self.matrix_calculate()
            
            A = []
            for i in range(self.R):
                A.append(self.orthogonal_projection(self.a_grad_u_grad_U(self.a_sto)[i]))
            A = np.array(A)
            self.Y += -self.dt * scipy.linalg.solve(self.matrix+ self.dt_a0_grad_U_grad_U(),A)
            
            #             self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix + self.dt_a0_grad_U_grad_U()),A)
            
            # reorthogonalize
            self.reorthogonalize()

            t  += self.dt
            count += 1
            
            self.mean_n.assign(self.mean) 
            for i in range(self.R):
                self.U_n[i].assign(self.U[i])
            
            if count % 1 == 0:
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

        
        

   #energy norm

    def energynorm(self):
        u = self.mean 
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        for i in range(1,self.sample_size):
            u = self.mean
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy
    
    def exl2norm(self):
        u = self.mean
        for j in range(self.R):
            u += self.U[j] * Constant(self.Y[j][0])
        form = u* u * dx
        for i in range(1,self.sample_size):
            u = self.mean
            for j in range(self.R):
                u += self.U[j] * Constant(self.Y[j][i])
            form += u * u * dx
        exl2 = assemble(form) / self.sample_size
        return np.sqrt(exl2)
    
    # def exl2norm(self):
    #     return norm(self.mean,'l2')
    
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
        # u = self.mean + self.U[0] * np.mean(self.Y[0]) + self.U[1] * np.mean(self.Y[1]) + self.U[2] * np.mean(self.Y[2])
        plot(self.mean)
        plt.show()
        
    

