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


# In[155]:



class Two_level_DLR:
    def __init__(self,dt,n_f,n_c,a,a_0,a_sto,Y_f = None,Y_c = None,delta_f = None,delta_c = None,YU = None,R_f = 3,R_c=5,sample_size = 50,mesh_type='1D'):
        self.dt = dt  # time step
        self.n_f = n_f  # number of mesh
        self.h_f = 1 / n_f  # element size
        self.n_c = n_c  # number of mesh
        self.h_c = 1 / n_c  # element size
        self.R_f = R_f  # rank of DLR
        self.R_c = R_c # rank of DLR
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh_f = UnitIntervalMesh(self.n_f)
            self.mesh_c = UnitIntervalMesh(self.n_c)
        elif mesh_type == '2D':
            self.mesh_f = UnitSquareMesh(self.n_f, self.n_f)
            self.mesh_c = UnitSquareMesh(self.n_c, self.n_c)
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V_f = FunctionSpace(self.mesh_f,'P',1)
        self.V_c = FunctionSpace(self.mesh_c,'P',1)
        
        ## Define boundary condition
        self.bc_f = DirichletBC(self.V_f, Constant(0), 'on_boundary')
        self.bc_c = DirichletBC(self.V_c, Constant(0), 'on_boundary')

        # Mass matrix
        self.mass_matrix_f = self._assemble_mass_matrix(self.V_f)
        self.mass_matrix_c = self._assemble_mass_matrix(self.V_c)

        ## Initialize functions 
    
        if YU == None:  
            # stochatic basis functions    
                
            self.Y_f = Y_f #(R_f,sample_size)
            self.Y_f = np.array(self.Y_f)
            self.Y_c = Y_c #(R_c,sample_size)
            self.Y_c = np.array(self.Y_c)
            self.Y_c_n = copy.deepcopy(self.Y_c)

            
            # Deterministic basis functions
            self.delta_f = [interpolate(df_i, self.V_f) for df_i in delta_f]
            self.delta_f_n = [interpolate(df_i, self.V_f) for df_i in delta_f]
            self.delta_c = [interpolate(dc_i, self.V_c) for dc_i in delta_c]
            self.delta_c_n = [interpolate(dc_i, self.V_c) for dc_i in delta_c]

            self.reorthogonalize_f()
            self.reorthogonalize_c()
        
            
            for i in range(self.R_f):
                self.delta_f_n[i].assign(self.delta_f[i])
            for i in range(self.R_c):
                self.delta_c_n[i].assign(self.delta_c[i])
            self.Y_c_n = copy.deepcopy(self.Y_c)
            
        else:
            self.delta_f = [Function(self.V_f) for i in range(self.R_f)]
            self.delta_f_n = [Function(self.V_f) for i in range(self.R_f)]
            self.delta_c = [Function(self.V_c) for i in range(self.R_c)]
            self.delta_c_n = [Function(self.V_c) for i in range(self.R_c)]
            v2d = vertex_to_dof_map(self.V_c)
            UY  = np.array([interpolate(YU[i],self.V_c).vector()[v2d] for i in range(self.sample_size)]).T
            Matrix = UY.T @ self.mass_matrix_c @ UY
            U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
            Vt_reduced = Vt[:self.R_c, :]
            self.Y_c = Vt_reduced  
            U_vectors = (Vt @ UY.T)[:self.R_c, :]
            self.Y_c  *= np.sqrt(self.sample_size)
            self.Y_c_n  = copy.deepcopy(self.Y_c)
            for i in range(self.R_c):
                self.delta_c[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                self.delta_c_n[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
            
            v2d = vertex_to_dof_map(self.V_f)
            UY  = np.array([interpolate(YU[i],self.V_f).vector()[v2d] for i in range(self.sample_size)]).T
            # UY2 = np.array([interpolate(interpolate(YU[i],self.V_c),self.V_f).vector()[v2d] for i in range(self.sample_size)]).T
            UY2 = np.array([interpolate(self.delta_c[i],self.V_f).vector()[v2d] for i in range(self.R_c)]).T @ self.Y_c
            UY = UY - UY2
            Matrix = UY.T @ self.mass_matrix_f @ UY
            U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
            Vt_reduced = Vt[:self.R_f, :]
            self.Y_f = Vt_reduced 
            U_vectors = (Vt @ UY.T)[:self.R_f, :]
            self.Y_f  *= np.sqrt(self.sample_size)
            for i in range(self.R_f):
                self.delta_f[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                self.delta_f_n[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                

        ## set coefficient
        self.a_0 = a_0
        self.a_sto = a_sto
        self.a = a 
        

        #M_i,j = <U_i,U_j>
        self.matrix_f = np.zeros((R_f,R_f)) 
        self.matrix_c = np.zeros((R_c,R_c)) 

        

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
    def matrix_calculate_f(self):
        v2d = vertex_to_dof_map(self.V_f)
        for i in range(self.R_f):
            for j in range(i, self.R_f):
                dof_i = self.delta_f[i].vector()[v2d]
                dof_j = self.delta_f[j].vector()[v2d]
                value = np.dot(dof_i, np.dot(self.mass_matrix_f, dof_j))
                self.matrix_f[i][j] = value
                self.matrix_f[j][i] = value

    def matrix_calculate_c(self):
        v2d = vertex_to_dof_map(self.V_c)
        for i in range(self.R_c):
            for j in range(i, self.R_c):
                dof_i = self.delta_c[i].vector()[v2d]
                dof_j = self.delta_c[j].vector()[v2d]
                value = np.dot(dof_i, np.dot(self.mass_matrix_c, dof_j))
                self.matrix_c[i][j] = value
                self.matrix_c[j][i] = value

                
                
    ## subfunctions for calculating dinamics
    
    def _build_function_f(self, i):
        func = Constant(0)
        for j in range(self.R_f):
            func += self.delta_f_n[j] * Constant(self.Y_f[j, i])
        for j in range(self.R_c):
            func += interpolate(self.delta_c_n[j], self.V_f) * Constant(self.Y_c_n[j, i])
        return func

    def _build_function_c(self, i):
        func = Constant(0)
        for j in range(self.R_c):
            func += self.delta_c_n[j] * Constant(self.Y_c_n[j, i])
        return func

    def E_a_grad_u_Y(self, a, Y, cf):
        if cf == "f":
            grad_list = [grad(self._build_function_f(i)) for i in range(self.sample_size)]
            ans = a[0] * grad_list[0] * Y[0]
            for i in range(1, self.sample_size):
                ans += a[i] * grad_list[i] * Y[i]
            ans /= self.sample_size
            ans = project(ans, VectorFunctionSpace(self.mesh_f, 'P', 1))

            grad_list2 = [grad(self._build_function_c(i)) for i in range(self.sample_size)]
            ans2 = a[0] * grad_list2[0] * Y[0]
            for i in range(1, self.sample_size):
                ans2 += a[i] * grad_list2[i] * Y[i]
            ans2 /= self.sample_size
            ans2 = project(project(ans2, VectorFunctionSpace(self.mesh_c, 'P', 1)),VectorFunctionSpace(self.mesh_f, 'P', 1))
            return ans - ans2

        elif cf == "c":
            grad_list = [grad(self._build_function_c(i)) for i in range(self.sample_size)]
            ans = a[0] * grad_list[0] * Y[0]
            for i in range(1, self.sample_size):
                ans += a[i] * grad_list[i] * Y[i]
            ans /= self.sample_size
            ans = project(ans, VectorFunctionSpace(self.mesh_c, 'P', 1))
            return ans

        else:
            raise ValueError("Invalid level_type. Use 'f' or 'c'.")

    
    
    def a_grad_u_grad_U(self,a,cf):
        if cf == "f":     
            ans = np.zeros((self.R_f,self.sample_size)) 
            u1 = [self._build_function_f(i) for i in range(self.sample_size)]
            u2 = [self._build_function_c(i) for i in range(self.sample_size)]
            
            for i in range(self.R_f):
                for j in range(self.sample_size):
                    a_grad_u_1 = project(a[j] * grad(u1[j]),VectorFunctionSpace(self.mesh_f, 'P', 1))
                    a_grad_u_2 = project(project(a[j] * grad(u2[j]),VectorFunctionSpace(self.mesh_c, 'P', 1)),VectorFunctionSpace(self.mesh_f, 'P', 1))
                    ans[i][j] = assemble(dot(a_grad_u_1 - a_grad_u_2 ,grad(self.delta_f[i])) * dx)
            return ans
        
        elif cf == "c":     
            ans = np.zeros((self.R_c,self.sample_size)) 
            u = [self._build_function_c(i) for i in range(self.sample_size)]
            
            for i in range(self.R_c):
                for j in range(self.sample_size):
                    a_grad_u = project(a[j] * grad(u[j]),VectorFunctionSpace(self.mesh_c, 'P', 1))
                    ans[i][j] =  assemble(dot(a_grad_u,grad(self.delta_c[i])) * dx)
            return ans
        
        else: 
            raise ValueError("Invalid level_type. Use 'f' or 'c'.")
    
  
                             
                          
    def orthogonal_projection_f(self,v):
        ortho = np.inner(v,self.Y_f[0]) / self.sample_size * self.Y_f[0]
        for i in range(1,self.R_f):
            ortho += np.inner(v,self.Y_f[i]) / self.sample_size * self.Y_f[i]
        return v - ortho
    
    def orthogonal_projection_c(self,v):
        ortho = np.inner(v,self.Y_c[0]) / self.sample_size * self.Y_c[0]
        for i in range(1,self.R_c):
            ortho += np.inner(v,self.Y_c[i]) / self.sample_size * self.Y_c[i]
        return v - ortho
    


    def reorthogonalize_f(self):
        Q, _ = np.linalg.qr(np.transpose(self.Y_f))
        self.Y_f =  np.sqrt(self.sample_size) * np.transpose(Q)
        vectors = []
        for i in range(self.R_f):
            vectors.append(self.delta_f[i].vector()[:])
        vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
        for i in range(self.R_f):
            self.delta_f[i].vector()[:] = vectors[i]
        
    def reorthogonalize_c(self):
        Q, _ = np.linalg.qr(np.transpose(self.Y_c))
        self.Y_c =  np.sqrt(self.sample_size) * np.transpose(Q)
        vectors = []
        for i in range(self.R_c):
            vectors.append(self.delta_c[i].vector()[:])
        vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
        for i in range(self.R_c):
            self.delta_c[i].vector()[:] = vectors[i]


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
        
        for i in range(self.R_f):
            self.delta_f[i] = TrialFunction(self.V_f)
        v_f = TestFunction(self.V_f)
        
        for i in range(self.R_c):
            self.delta_c[i] = TrialFunction(self.V_c)
        v_c = TestFunction(self.V_c)

        lhs_f = []
        rhs_f = []
        lhs_c = []
        rhs_c = []
        
        for i in range(self.R_c):
            a_i = self.delta_c[i] * v_c * dx
            L_i = self.delta_c_n[i] * v_c* dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y_c[i],cf= "c"), grad(v_c)) * dx
            lhs_c.append(a_i)
            rhs_c.append(L_i)
           
        for i in range(self.R_c):
            self.delta_c[i] = Function(self.V_c)
        
        for i in range(self.R_f):
            a_i = self.delta_f[i] * v_f * dx
            L_i = self.delta_f_n[i] * v_f * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y_f[i],cf= "f"), grad(v_f)) * dx
            lhs_f.append(a_i)
            rhs_f.append(L_i)
           
        for i in range(self.R_f):
            self.delta_f[i] = Function(self.V_f)
        
        while t < end:           
            # Compute solution
            for i in range(self.R_c):
                solve(lhs_c[i]==rhs_c[i],self.delta_c[i],self.bc_c)
         
            self.matrix_calculate_c()

            A_c = [self.orthogonal_projection_c(self.a_grad_u_grad_U(self.a, cf="c")[i]) for i in range(self.R_c)]
            A_c = np.array(A_c)
            det = np.linalg.det(self.matrix_c)
            if np.isclose(det, 0):
                self.Y_c += -self.dt * scipy.linalg.lstsq(self.matrix_c,A_c)[0]
            else:
                self.Y_c += -self.dt * scipy.linalg.solve(self.matrix_c,A_c)
                #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
                
            
            # reorthogonalize
            self.reorthogonalize_c()
            
            
            
            for i in range(self.R_f):
                solve(lhs_f[i]==rhs_f[i],self.delta_f[i],self.bc_f)
         
            self.matrix_calculate_f()
            A_f = [self.orthogonal_projection_f(self.a_grad_u_grad_U(self.a, cf="f")[i]) for i in range(self.R_f)]
            A_f = np.array(A_f)
            det = np.linalg.det(self.matrix_f)
            if np.isclose(det, 0):
                self.Y_f += -self.dt * scipy.linalg.lstsq(self.matrix_f,A_f)[0]
            else:
                self.Y_f += -self.dt * scipy.linalg.solve(self.matrix_f,A_f)
                #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
                
            
            # reorthogonalize
            self.reorthogonalize_f()

            t  += self.dt
            count += 1
            
            for i in range(self.R_c):
                self.delta_c_n[i].assign(self.delta_c[i])
            for i in range(self.R_f):
                self.delta_f_n[i].assign(self.delta_f[i])
            self.Y_c_n = copy.deepcopy(self.Y_c)
           
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

    # Energy norm
    def energynorm(self):
        u = Function(self.V_f)
        for j in range(self.R_f):
            u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][0]
        for j in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][0]
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        for i in range(1, self.sample_size):
            u = Function(self.V_f)
            for j in range(self.R_f):
                u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][i]
            for j in range(self.R_c):
                u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][i]
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy

    def exl2norm(self):
        u = Function(self.V_f)
        for j in range(self.R_f):
            u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][0]
        for j in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][0]
        form = u * u * dx
        for i in range(1, self.sample_size):
            u = Function(self.V_f)
            for j in range(self.R_f):
                u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][i]
            for j in range(self.R_c):
                u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][i]
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
        u = Function(self.V_f)
        for i in range(self.R_f):
            u.vector()[:] += self.delta_f[i].vector()[:] * np.mean(self.Y_f[i])
        for i in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[i], self.V_f).vector()[:] * np.mean(self.Y_c[i])
        plot(u)
        plt.show()
        
class Two_level_DLR2:
    def __init__(self,dt,n_f,n_c,a,a_0,a_sto,Y_f = None,Y_c = None,delta_f = None,delta_c = None,YU = None,R_f = 3,R_c=5,sample_size = 50,mesh_type='1D'):
        self.dt = dt  # time step
        self.n_f = n_f  # number of mesh
        self.h_f = 1 / n_f  # element size
        self.n_c = n_c  # number of mesh
        self.h_c = 1 / n_c  # element size
        self.R_f = R_f  # rank of DLR
        self.R_c = R_c # rank of DLR
        self.sample_size = sample_size  # stochastic discretization
        
        if mesh_type == '1D':
            self.mesh_f = UnitIntervalMesh(self.n_f)
            self.mesh_c = UnitIntervalMesh(self.n_c)
        elif mesh_type == '2D':
            self.mesh_f = UnitSquareMesh(self.n_f, self.n_f)
            self.mesh_c = UnitSquareMesh(self.n_c, self.n_c)
        else:
            raise ValueError("Invalid mesh_type. Use '1D' or '2D'.")

        self.V_f = FunctionSpace(self.mesh_f,'P',1)
        self.V_c = FunctionSpace(self.mesh_c,'P',1)
        
        ## Define boundary condition
        self.bc_f = DirichletBC(self.V_f, Constant(0), 'on_boundary')
        self.bc_c = DirichletBC(self.V_c, Constant(0), 'on_boundary')

        # Mass matrix
        self.mass_matrix_f = self._assemble_mass_matrix(self.V_f)
        self.mass_matrix_c = self._assemble_mass_matrix(self.V_c)

        ## Initialize functions 
    
        if YU == None:  
            # stochatic basis functions    
                
            self.Y_f = Y_f #(R_f,sample_size)
            self.Y_f = np.array(self.Y_f)
            self.Y_c = Y_c #(R_c,sample_size)
            self.Y_c = np.array(self.Y_c)
            self.Y_c_n = copy.deepcopy(self.Y_c)

            
            # Deterministic basis functions
            self.delta_f = [interpolate(df_i, self.V_f) for df_i in delta_f]
            self.delta_f_n = [interpolate(df_i, self.V_f) for df_i in delta_f]
            self.delta_c = [interpolate(dc_i, self.V_c) for dc_i in delta_c]
            self.delta_c_n = [interpolate(dc_i, self.V_c) for dc_i in delta_c]

            self.reorthogonalize_f()
            self.reorthogonalize_c()
        
            
            for i in range(self.R_f):
                self.delta_f_n[i].assign(self.delta_f[i])
            for i in range(self.R_c):
                self.delta_c_n[i].assign(self.delta_c[i])
            self.Y_c_n = copy.deepcopy(self.Y_c)
            
        else:
            self.delta_f = [Function(self.V_f) for i in range(self.R_f)]
            self.delta_f_n = [Function(self.V_f) for i in range(self.R_f)]
            self.delta_c = [Function(self.V_c) for i in range(self.R_c)]
            self.delta_c_n = [Function(self.V_c) for i in range(self.R_c)]
            v2d = vertex_to_dof_map(self.V_c)
            UY  = np.array([interpolate(YU[i],self.V_c).vector()[v2d] for i in range(self.sample_size)]).T
            Matrix = UY.T @ self.mass_matrix_c @ UY
            U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
            Vt_reduced = Vt[:self.R_c, :]
            self.Y_c = Vt_reduced  
            U_vectors = (Vt @ UY.T)[:self.R_c, :]
            self.Y_c  *= np.sqrt(self.sample_size)
            self.Y_c_n  = copy.deepcopy(self.Y_c)
            for i in range(self.R_c):
                self.delta_c[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                self.delta_c_n[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
            
            v2d = vertex_to_dof_map(self.V_f)
            UY  = np.array([interpolate(YU[i],self.V_f).vector()[v2d] for i in range(self.sample_size)]).T
            # UY2 = np.array([interpolate(interpolate(YU[i],self.V_c),self.V_f).vector()[v2d] for i in range(self.sample_size)]).T
            UY2 = np.array([interpolate(self.delta_c[i],self.V_f).vector()[v2d] for i in range(self.R_c)]).T @ self.Y_c
            UY = UY - UY2
            Matrix = UY.T @ self.mass_matrix_f @ UY
            U, S, Vt = np.linalg.svd(Matrix, full_matrices=False,hermitian=True)
            Vt_reduced = Vt[:self.R_f, :]
            self.Y_f = Vt_reduced 
            U_vectors = (Vt @ UY.T)[:self.R_f, :]
            self.Y_f  *= np.sqrt(self.sample_size)
            for i in range(self.R_f):
                self.delta_f[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                self.delta_f_n[i].vector()[v2d] = U_vectors[i] / np.sqrt(self.sample_size)
                

        ## set coefficient
        self.a_0 = a_0
        self.a_sto = a_sto
        self.a = a 
        

        #M_i,j = <U_i,U_j>
        self.matrix_f = np.zeros((R_f,R_f)) 
        self.matrix_c = np.zeros((R_c,R_c)) 

        

        ##for plot
        self.timelist = []
        self.energylist = []
        self.L2list= []

        self.build_c = [Constant(0)] * self.sample_size 
        self.build_f = [Constant(0)] * self.sample_size
        self.a_delta_u_list_c = [Constant(0)] * self.sample_size  
        self.a_delta_u_list = [Constant(0)] * self.sample_size  
        self._build_function_c()
        self._build_function_f()
        self.a_delta_u()

        
   # calculate Mass matrix
    def _assemble_mass_matrix(self, V):
            tri = TrialFunction(V)
            test = TestFunction(V)
            integral = tri * test * dx
            A = assemble(integral)
            return as_backend_type(A).mat().getValues(range(A.size(0)), range(A.size(1)))


    # calculate quadratures, M_i,j = <U_i,U_j>
    def matrix_calculate_f(self):
        v2d = vertex_to_dof_map(self.V_f)
        for i in range(self.R_f):
            for j in range(i, self.R_f):
                dof_i = self.delta_f[i].vector()[v2d]
                dof_j = self.delta_f[j].vector()[v2d]
                value = np.dot(dof_i, np.dot(self.mass_matrix_f, dof_j))
                self.matrix_f[i][j] = value
                self.matrix_f[j][i] = value

    def matrix_calculate_c(self):
        v2d = vertex_to_dof_map(self.V_c)
        for i in range(self.R_c):
            for j in range(i, self.R_c):
                dof_i = self.delta_c[i].vector()[v2d]
                dof_j = self.delta_c[j].vector()[v2d]
                value = np.dot(dof_i, np.dot(self.mass_matrix_c, dof_j))
                self.matrix_c[i][j] = value
                self.matrix_c[j][i] = value

                
                
    ## subfunctions for calculating dinamics
    
    def _build_function_f(self):
        for i in range(self.sample_size):
            func = Constant(0)
            for j in range(self.R_f):
                func += self.delta_f_n[j] * Constant(self.Y_f[j, i])
            for j in range(self.R_c):
                func += interpolate(self.delta_c_n[j], self.V_f) * Constant(self.Y_c_n[j, i])
            self.build_f[i] = func

    def _build_function_c(self):
        for i in range(self.sample_size):
            func = Constant(0)
            for j in range(self.R_c):
                func += self.delta_c_n[j] * Constant(self.Y_c_n[j, i])
            self.build_c[i] = func
    def a_delta_u(self):
        for i in range(self.sample_size):
            ans = TrialFunction(self.V_c)
            v_c = TestFunction(self.V_c)
            u = self.build_c[i]
            l = ans * v_c * dx
            r = - self.a[i] * dot(grad(u),grad(v_c)) * dx
            ans = Function(self.V_c)
            solve(l==r,ans,self.bc_c)
            self.a_delta_u_list_c[i] = ans
            self.a_delta_u_list[i] = interpolate(ans,self.V_f)
    
    def E_a_delta_u_Y(self,Y):
        ans = Constant(0)
        for i in range(self.sample_size):
            ans += self.a_delta_u_list[i] * Y[i]
        ans /= self.sample_size
        return ans     

    def E_a_grad_u_Y(self, a, Y, cf):
        if cf == "f":
            ans = a[0] * grad(self.build_f[0]) * Y[0]
            for i in range(1, self.sample_size):
                ans += a[i] * grad(self.build_f[i]) * Y[i]
            ans /= self.sample_size
            return ans 

        # elif cf == "c":
        #     ans = a[0] * grad(self.build_c[0]) * Y[0]
        #     for i in range(1, self.sample_size):
        #         ans += a[i] * grad(self.build_c[i]) * Y[i]
        #     ans /= self.sample_size
        #     return ans

        else:
            raise ValueError("Invalid level_type. Use 'f' or 'c'.")

    
    def a_grad_u_grad_U(self,a,cf):
        if cf == "f":     
            ans = np.zeros((self.R_f,self.sample_size)) 
            u1 = self.build_f
            
            for i in range(self.R_f):
                for j in range(self.sample_size):
                    a_grad_u_1 = a[j] * grad(u1[j])
                    ans[i][j] = assemble(dot(a_grad_u_1, grad(self.delta_f[i])) *dx + self.a_delta_u_list[j] * self.delta_f[i] * dx)
            return ans
        
        elif cf == "c":     
            ans = np.zeros((self.R_c,self.sample_size)) 
            u = self.build_c
            for i in range(self.R_c):
                for j in range(self.sample_size):
                    a_grad_u = a[j] * grad(u[j])
                    ans[i][j] =  assemble(dot(a_grad_u,grad(self.delta_c[i])) * dx)
            return ans
        
        else: 
            raise ValueError("Invalid level_type. Use 'f' or 'c'.")
    
  
                             
                          
    def orthogonal_projection_f(self,v):
        ortho = np.inner(v,self.Y_f[0]) / self.sample_size * self.Y_f[0]
        for i in range(1,self.R_f):
            ortho += np.inner(v,self.Y_f[i]) / self.sample_size * self.Y_f[i]
        return v - ortho
    
    def orthogonal_projection_c(self,v):
        ortho = np.inner(v,self.Y_c[0]) / self.sample_size * self.Y_c[0]
        for i in range(1,self.R_c):
            ortho += np.inner(v,self.Y_c[i]) / self.sample_size * self.Y_c[i]
        return v - ortho
    


    def reorthogonalize_f(self):
        Q, _ = np.linalg.qr(np.transpose(self.Y_f))
        self.Y_f =  np.sqrt(self.sample_size) * np.transpose(Q)
        vectors = []
        for i in range(self.R_f):
            vectors.append(self.delta_f[i].vector()[:])
        vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
        for i in range(self.R_f):
            self.delta_f[i].vector()[:] = vectors[i]
        
    def reorthogonalize_c(self):
        Q, _ = np.linalg.qr(np.transpose(self.Y_c))
        self.Y_c =  np.sqrt(self.sample_size) * np.transpose(Q)
        vectors = []
        for i in range(self.R_c):
            vectors.append(self.delta_c[i].vector()[:])
        vectors = np.matmul(_ /np.sqrt(self.sample_size),vectors)
        for i in range(self.R_c):
            self.delta_c[i].vector()[:] = vectors[i]


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
        
        for i in range(self.R_f):
            self.delta_f[i] = TrialFunction(self.V_f)
        v_f = TestFunction(self.V_f)
        
        # for i in range(self.R_c):
        #     self.delta_c[i] = TrialFunction(self.V_c)
        # v_c = TestFunction(self.V_c)

        lhs_f = []
        rhs_f = []
        # lhs_c = []
        # rhs_c = []
        
        # for i in range(self.R_c):
        #     a_i = self.delta_c[i] * v_c * dx
        #     L_i = self.delta_c_n[i] * v_c* dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y_c[i],cf= "c"), grad(v_c)) * dx
        #     lhs_c.append(a_i)
        #     rhs_c.append(L_i)
           
        # for i in range(self.R_c):
        #     self.delta_c[i] = Function(self.V_c)
        
        for i in range(self.R_f):
            a_i = self.delta_f[i] * v_f * dx
            L_i = self.delta_f_n[i] * v_f * dx - self.dt * dot(self.E_a_grad_u_Y(self.a,self.Y_f[i],cf= "f"), grad(v_f)) * dx - self.dt * self.E_a_delta_u_Y(self.Y_f[i]) * v_f * dx
            lhs_f.append(a_i)
            rhs_f.append(L_i)
           
        for i in range(self.R_f):
            self.delta_f[i] = Function(self.V_f)
        
        while t < end:
            self._build_function_c()
            self._build_function_f()
            self.a_delta_u()

            # Compute solution
            for i in range(self.R_c):
                diff = interpolate(Constant(0),self.V_c)
                for j in range(self.sample_size):
                    diff.vector()[:] += self.a_delta_u_list_c[j].vector()[:] * self.Y_c_n[i][j]
                diff.vector()[:] /= self.sample_size
                self.delta_c[i].vector()[:] += self.dt * diff.vector()[:]
            self.matrix_calculate_c()

            A_c = [self.orthogonal_projection_c(self.a_grad_u_grad_U(self.a, cf="c")[i]) for i in range(self.R_c)]
            A_c = np.array(A_c)
            det = np.linalg.det(self.matrix_c)
            if np.isclose(det, 0):
                self.Y_c += -self.dt * scipy.linalg.lstsq(self.matrix_c,A_c)[0]
            else:
                self.Y_c += -self.dt * scipy.linalg.solve(self.matrix_c,A_c)
                #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
                
            
            
            
            
            
            for i in range(self.R_f):
                solve(lhs_f[i]==rhs_f[i],self.delta_f[i],self.bc_f)
         
            self.matrix_calculate_f()
            A_f = [self.orthogonal_projection_f(self.a_grad_u_grad_U(self.a, cf="f")[i]) for i in range(self.R_f)]
            A_f = np.array(A_f)
            det = np.linalg.det(self.matrix_f)
            if np.isclose(det, 0):
                self.Y_f += -self.dt * scipy.linalg.lstsq(self.matrix_f,A_f)[0]
            else:
                self.Y_f += -self.dt * scipy.linalg.solve(self.matrix_f,A_f)
                #                 self.Y += -self.dt * np.matmul(scipy.linalg.inv(self.matrix),A)
                
            
            # reorthogonalize
            # reorthogonalize
            self.reorthogonalize_c()
            self.reorthogonalize_f()

            t  += self.dt
            count += 1
            
            for i in range(self.R_c):
                self.delta_c_n[i].assign(self.delta_c[i])
            for i in range(self.R_f):
                self.delta_f_n[i].assign(self.delta_f[i])
            self.Y_c_n = copy.deepcopy(self.Y_c)
           
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

    # Energy norm
    def energynorm(self):
        u = Function(self.V_f)
        for j in range(self.R_f):
            u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][0]
        for j in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][0]
        form = self.a[0] * inner(grad(u), grad(u)) * dx
        for i in range(1, self.sample_size):
            u = Function(self.V_f)
            for j in range(self.R_f):
                u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][i]
            for j in range(self.R_c):
                u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][i]
            form += self.a[i] * inner(grad(u), grad(u)) * dx
        energy = assemble(form) / self.sample_size
        return energy

    def exl2norm(self):
        u = Function(self.V_f)
        for j in range(self.R_f):
            u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][0]
        for j in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][0]
        form = u * u * dx
        for i in range(1, self.sample_size):
            u = Function(self.V_f)
            for j in range(self.R_f):
                u.vector()[:] += self.delta_f[j].vector()[:] * self.Y_f[j][i]
            for j in range(self.R_c):
                u.vector()[:] += interpolate(self.delta_c[j], self.V_f).vector()[:] * self.Y_c[j][i]
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
        u = Function(self.V_f)
        for i in range(self.R_f):
            u.vector()[:] += self.delta_f[i].vector()[:] * np.mean(self.Y_f[i])
        for i in range(self.R_c):
            u.vector()[:] += interpolate(self.delta_c[i], self.V_f).vector()[:] * np.mean(self.Y_c[i])
        plot(u)
        plt.show()
        
