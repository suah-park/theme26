"""Periodic-only extraction of Yang's SinglePhase class. MIT, see LICENSE.

NOT a byte-identical copy of the source and NOT certified as the paper code.
See UPSTREAM.json and docs/evaluation_fp64.md. Removed pressure/velocity BC,
VTK, unused imports, sparse fields and global setup; arrays enter directly.

Single formula: tau=3*nu+0.5, with fixed force coefficients 3 and 9.
Tau matches Yang Eq.7 and the standalone upstream implementation; force
prefactors remain a separately documented correction to the class. No fitted
789 mD scaling and no alternative runtime formula are provided.

v0.3.0 fixes all real-valued fields/constants to FP64 (the v0.2.2 FP64 path).
No collision, forcing, streaming or macroscopic formula is changed.
This is still a derivative, not a return to unmodified author code.
"""
import numpy as np
import taichi as ti
from .lattice import E, W, M, LR
from .formulation import tau_from_nu, FORCE_LINEAR, FORCE_QUADRATIC

@ti.data_oriented
class LB3D_Solver_Single_Phase:
    def __init__(self, nx, ny, nz, nu, force, precision="float64"):
        if precision != "float64":
            raise ValueError("This final solver is float64 only")
        self.fp = ti.f64
        np_fp = np.float64
        self.nx, self.ny, self.nz = nx, ny, nz
        self.fx, self.fy, self.fz = force
        self.tau_f = tau_from_nu(nu)
        sv = 1.0/self.tau_f
        other = 8.0*(2.0-sv)/(8.0-sv)
        self.a_force = FORCE_LINEAR
        self.b_force = FORCE_QUADRATIC
        self.f = ti.Vector.field(19,self.fp,shape=(nx,ny,nz))
        self.F = ti.Vector.field(19,self.fp,shape=(nx,ny,nz))
        self.rho = ti.field(self.fp,shape=(nx,ny,nz))
        self.v = ti.Vector.field(3,self.fp,shape=(nx,ny,nz))
        self.solid = ti.field(ti.i8,shape=(nx,ny,nz))
        self.e = ti.Vector.field(3,ti.i32,shape=19)
        self.e_f = ti.Vector.field(3,self.fp,shape=19)
        self.w = ti.field(self.fp,shape=19)
        self.M = ti.Matrix.field(19,19,self.fp,shape=())
        self.inv_M = ti.Matrix.field(19,19,self.fp,shape=())
        self.S_dig = ti.Vector.field(19,self.fp,shape=())
        self.LR = LR
        self.e.from_numpy(E); self.e_f.from_numpy(E.astype(np_fp))
        self.w.from_numpy(W.astype(np_fp))
        self.M.from_numpy(M.astype(np_fp))
        self.inv_M.from_numpy(np.linalg.inv(M).astype(np_fp))
        self.S_dig.from_numpy(np.asarray([0,sv,sv,0,other,0,other,0,other,sv,sv,sv,sv,sv,sv,sv,other,other,other], dtype=np_fp))

    @ti.func
    def feq(self,k,rho_local,u):
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        return self.w[k]*rho_local*(1.0+3.0*eu+4.5*eu*eu-1.5*uv)

    @ti.kernel
    def init(self):
        for i,j,k in self.solid:
            self.rho[i,j,k] = 1.0
            self.v[i,j,k] = ti.Vector([0.0,0.0,0.0])
            for s in ti.static(range(19)):
                self.f[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])
                self.F[i,j,k][s] = self.feq(s,1.0,self.v[i,j,k])

    @ti.func
    def meq_vec(self,rho_local,u):
        # Author's incompressible-moment convention is retained, including u
        # (not rho*u) in conserved moments. Low density variation is required.
        out = ti.Vector([0.0]*19)
        out[0]=rho_local; out[3]=u[0]; out[5]=u[1]; out[7]=u[2]
        out[1]=u.dot(u); out[9]=2*u.x*u.x-u.y*u.y-u.z*u.z
        out[11]=u.y*u.y-u.z*u.z
        out[13]=u.x*u.y; out[14]=u.y*u.z; out[15]=u.x*u.z
        return out

    @ti.func
    def cal_local_force(self,i,j,k):
        return ti.Vector([self.fx,self.fy,self.fz])

    @ti.kernel
    def colission(self):
        for i,j,k in self.rho:
            if self.solid[i,j,k] == 0:
                m_temp = self.M[None]@self.F[i,j,k]
                meq = self.meq_vec(self.rho[i,j,k],self.v[i,j,k])
                m_temp -= self.S_dig[None]*(m_temp-meq)
                f = self.cal_local_force(i,j,k)
                for s in ti.static(range(19)):
                    f_guo = 0.0
                    for l in ti.static(range(19)):
                        f_guo += self.w[l]*(self.a_force*(self.e_f[l]-self.v[i,j,k]).dot(f)+self.b_force*(self.e_f[l].dot(self.v[i,j,k])*(self.e_f[l].dot(f))))*self.M[None][s,l]
                    m_temp[s] += (1-0.5*self.S_dig[None][s])*f_guo
                self.f[i,j,k] = self.inv_M[None]@m_temp

    @ti.func
    def periodic_index(self,i):
        iout = i
        if i[0]<0: iout[0]=self.nx-1
        if i[0]>self.nx-1: iout[0]=0
        if i[1]<0: iout[1]=self.ny-1
        if i[1]>self.ny-1: iout[1]=0
        if i[2]<0: iout[2]=self.nz-1
        if i[2]>self.nz-1: iout[2]=0
        return iout

    @ti.kernel
    def streaming1(self):
        for i in ti.grouped(self.rho):
            if self.solid[i] == 0:
                for s in ti.static(range(19)):
                    ip = self.periodic_index(i+self.e[s])
                    if self.solid[ip] == 0:
                        self.F[ip][s] = self.f[i][s]
                    else:
                        self.F[i][self.LR[s]] = self.f[i][s]

    @ti.kernel
    def streaming3(self):
        for i in ti.grouped(self.rho):
            if self.solid[i] == 0:
                self.rho[i] = 0.0
                self.v[i] = ti.Vector([0.0,0.0,0.0])
                self.f[i] = self.F[i]
                self.rho[i] += self.f[i].sum()
                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s]*self.f[i][s]
                f = self.cal_local_force(i.x,i.y,i.z)
                self.v[i] /= self.rho[i]
                self.v[i] += (f/2)/self.rho[i]
            else:
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0.0,0.0,0.0])

    def step(self):
        self.colission()
        self.streaming1()
        self.streaming3()
