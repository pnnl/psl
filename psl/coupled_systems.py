"""
"""

import numpy as np
from scipy.sparse import coo_matrix
from psl.autonomous import ODE_Autonomous
from psl.nonautonomous import ODE_NonAutonomous
from psl.perturb import Periodic, Sawtooth
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm

def multidim(is_autonomous):
    def decorator(ode):
        ode._simulate = ode.simulate
        
        if is_autonomous:
            def sim_dec(self, ninit=None, nsim=None, Time=None, ts=None, x0=None, show_progress=False):
                x0 = x0 if x0 is not None else self.x0
                shape = x0.shape
                out = ode._simulate(self, ninit, nsim, Time, ts, x0.ravel(), show_progress)
                out['Y'].shape = (-1,) + shape
                out['X'].shape = (-1,) + shape
                return out
        else:
            def sim_dec(self, U=None, ninit=None, nsim=None, Time=None, ts=None, x0=None, show_progress=False):
                x0 = x0 if x0 is not None else self.x0
                shape = x0.shape
                out = ode._simulate(self, U, ninit, nsim, Time, ts, x0.ravel(), show_progress)
                out['Y'].shape = (-1,) + shape
                out['X'].shape = (-1,) + shape
                return out

        ode._equations = ode.equations
        def eq_dec(self, x, *args, **kwargs):
            shape = (self.nx, -1)
            x_new = x.reshape(shape)
            dx = ode._equations(self, x_new, *args, **kwargs)
            return dx.ravel()
        
        ode.simulate = sim_dec
        ode.equations = eq_dec
        return ode
    return decorator
           
class Coupled_ODE(ODE_Autonomous):
    def __init__(self, nsim=1001, ninit=0, ts=0.1, adj=None, nx=1, seed=59):
        super().__init__(nsim, ninit, ts, seed)
        self.nx=nx
        if adj is None:
            self.adj = np.ones((nx,nx))
            np.fill_diagonal(self.adj, 0)
            self.adj_list = np.stack(np.nonzero(self.adj))
        elif adj.shape[0] == 2: #(2, edges)  (i,j) adj_list[:, k]
            self.adj_list = adj
            self.adj = coo_matrix((np.ones(adj.shape[1]), (adj[0],adj[1])), shape=(nx,nx))
        else:
            self.adj = adj
            self.adj_list = np.stack(np.nonzero(self.adj))
    
    def message_passing(self,receivers, senders, t):
        """Stub function for message passing operation
        Systems that inherit should compute interactions between corresponding rows of receivers and senders
        :return messages from senders to receivers np.array of shape(len(receivers), **)

        :param receivers: _description_
        :param senders: _description_
        :param t: _description_
        """
        pass
    
    def equations(self, x, t):
        messages = self.message_passing(x[self.adj_list[0]], x[self.adj_list[1]], t)
        dx = np.zeros_like(x)
        np.add.at(dx, self.adj_list[0], messages)
        return dx
    
class Coupled_NonAutonomous(ODE_NonAutonomous):
    def __init__(self, nsim=1001, ninit=0, ts=0.1, adj=None, nx=1, seed=59):
        super().__init__(nsim, ninit, ts, seed)
        self.nx=nx
        if adj is None:
            self.adj = np.ones((nx,nx))
            np.fill_diagonal(self.adj, 0)
            self.adj_list = np.stack(np.nonzero(self.adj))
        elif adj.shape[0] == 2:
            self.adj_list = adj
            self.adj = coo_matrix((np.ones(adj.shape[1]), (adj[0],adj[1])), shape=(nx,nx))
        else:
            self.adj = adj
            self.adj_list = np.stack(np.nonzero(self.adj))
        self.U = list()
            
    def equations(self, x, t, u):
        messages = self.message_passing(x[self.adj_list[0]], x[self.adj_list[1]], t, u)
        dx = np.zeros_like(x)
        np.add.at(dx, self.adj_list[0], messages)
        return dx
    
    def message_passing(self, receivers, senders, t, u):
        """Stub function for message passing operation
        Systems that inherit should compute interactions between corresponding rows of receivers and senders: (receivers[i], senders[i]) are coupled
        :return messages from senders to receivers np.array of shape(len(receivers), **)

        :param receivers: Receiving agents 
        :param senders: Sending agents
        :param t: time step
        :param u: Control Variables
        """
        pass

 
 
class RC_Network(Coupled_NonAutonomous):
    def __init__(self, R = None, C = None, U=None, nsim=1001, ninit=0, ts=0.1, adj=None, nx=2, seed=59):
        """_summary_

        :param R: [float, np.array], Coupled Resistances
        :param C:[float, np.array]. Room Capacitance
        :param nsim: [int], length of simulation, defaults to 1001
        :param ninit: [int], starting time of simulation, defaults to 0
        :param ts: [float], rate of sampling, defaults to 0.1
        :param adj: [np.array, shape=(2,*) or (nx,nx)], adjacency list or matrix, defaults to None
        :param nx: (int), number of nodes, defaults to 1
        :param seed: seed for random number generator, defaults to 59
        """
        super().__init__(nsim, ninit, ts, adj, nx, seed)
        self.R = R if R is not None else self.get_R(
            self.adj_list, amax=20, amin=5, symmetric=True)
        self.C = C if C is not None else self.get_C(nx)
        self.U = U if U is not None else self.get_U(nsim, nx)
        self.R_ext = self.get_R(np.tile(np.arange(nx),(2,1)), amax=15, symmetric=False)
        
    def get_U(self, nsim, nx, periods = None):
        if periods is None:
            periods = int(np.ceil(nsim / 2000))
        global_source = Periodic(nsim=nsim, xmin=275.0, xmax=285.0)
        ind_sources = Periodic(nx=nx, nsim=nsim, form = 'square', numPeriods=periods, xmax=294, xmin = 0)
        return np.hstack([global_source,ind_sources])

    def get_R(self, adj_list, Rval=3.5, amax=20, amin=0, symmetric=True):
        #Default Rval is fiberglass insulation
        num = adj_list.shape[1]
        m2 = np.random.rand(num) * (amax-amin) + amin #surface area
        if symmetric:
            edge_map = {(i,j) : idx for idx, (i,j) in enumerate(adj_list.T)}
            edge_map = [edge_map[(d, s)] for (s,d) in adj_list.T]
            m2 = (m2 + m2[edge_map]) / 2.0  
        m2 = np.maximum(m2, 0.0000001)
        R = Rval / m2
        return R

    def get_C(self, num=1):
        C_air = 700
        d_air = 1.2
        v_max = 10.0 * 10.0 * 5.0
        v_min = 3.0 * 3.0 * 3.0
        V = np.random.rand(num) * (v_max - v_min) + v_min
        return C_air * d_air * V / 100.0
        
    def message_passing(self, receivers, senders, t, u):
        R = self.R        
        if type(self.C) is np.ndarray:
            C = self.C[self.adj_list[0]]
        else:
            C = self.C
        messages = (1.0 / (R*C)) * (senders - receivers)
        return messages
    
    def equations(self, x, t, u):
        external_source = u[0]
        internal_sources = u[1:]
    
        dx = np.zeros_like(x)
                
        #Internal heat transfer
        messages = self.message_passing(x[self.adj_list[0]], x[self.adj_list[1]], t, u)
        np.add.at(dx, self.adj_list[0], messages)
        
        #Outside heat transfer
        deltas = external_source - x
        dx += (1.0 / (self.R_ext * self.C)) * deltas
        
        #Internal heat sources
        deltas = internal_sources - x
        deltas[internal_sources==0] = 0
        dx += (1.0 / self.C) * deltas
        return dx
  
@multidim(True)
class Gravitational_System(Coupled_ODE):
    mass_idx = [0]
    pos_idx = [1,2]
    vel_idx = [3,4]
        
    def __init__(self, G=6.67e-11, nsim=1001, ninit=0, ts=0.1, adj=None, nx=1, seed=59):
        super().__init__(nsim, ninit, ts, adj, nx, seed)
        self.G = G

    def message_passing(self, receivers, senders, t):
        #Assumes rows of the form [mass, x_pos, y_pos, x_vel, y_vel]
        vectors = senders[:, self.pos_idx] - receivers[:, self.pos_idx]
        distances2 = np.sum(vectors**2, axis=1, keepdims=True)
        forces = self.G * ((senders[:, self.mass_idx]) / distances2) #divided by receiver mass to get accelerations
        force_vectors = forces * (vectors / np.linalg.norm(vectors,axis=1,keepdims=True))
        return force_vectors
    
    def equations(self, x ,t):
        #Assumes x is  in the form (mass, x_pos, y_pos, x_vel, y_vel)
        messages = self.message_passing(x[self.adj_list[0]], x[self.adj_list[1]], t)
        dx = np.zeros_like(x)
        acc = np.zeros((x.shape[0],2))
        np.add.at(acc, self.adj_list[0], messages)
        dx[:, self.vel_idx] = acc
        dx[:, self.pos_idx] = x[:, self.vel_idx]
        return dx

@multidim(True)
class Boids(Coupled_ODE):
    pos_idx = [0,1]
    vel_idx = [2,3]
    
    def __init__(self, coherence=0.05, separation=0.01, alignment=0.05, avoidance_range = 0.2, visual_range=None, nsim=1001, ninit=0, ts=0.1, nx=1, seed=59):
          super().__init__(nsim, ninit, ts, None, nx, seed)
          self.coherence = coherence
          self.separation = separation
          self.alignment = alignment
          self.avoidance_range = avoidance_range
          self.visual_range = visual_range if visual_range is None else visual_range**2
          
          self.max_speed = 0.03
          self.max_acc = 0.005
          
    def message_passing(self, receivers, senders, t):
        return senders-receivers

    def normalize_max(self, x, length):
        idx = np.sum(x*x,1) > length**2
        l2 = np.linalg.norm(x[idx], ord=2, axis=1, keepdims=True)
        x[idx] = (x[idx]/l2)*length
        return x

    def equations(self, x, t):
        pos = x[:, self.pos_idx]
        vel = x[:, self.vel_idx]
        
        #Contain Speed
        vel = self.normalize_max(vel, self.max_speed)
        
        #calculate adjacencies
        x2y2 = np.sum(pos*pos,1,keepdims=True)
        x2y2 = x2y2 + x2y2.T
        xy = pos.dot(pos.T)
        d2 = x2y2 - 2 * xy
        if self.visual_range is None:
            r = self.adj_list[0]
            s = self.adj_list[1]
            n=len(pos)-1
        else:
            adj = np.argwhere(d2 < self.visual_range)
            r=adj[:,0]
            s=adj[:,1]
            u, n = np.unique(r, return_counts=True)
            if len(u) < len(pos):
                return np.zeros_like(x)
            n = np.maximum(n[:,np.newaxis] -1, 1)
        
        #Cohesion
        r1 = np.zeros_like(pos)
        messages = self.message_passing(pos[r], pos[s], t)
        np.add.at(r1, r, messages)
        r1 = self.coherence * r1 / n
        
        #Alignment
        r2 = np.zeros_like(vel)
        messages = self.message_passing(vel[r], vel[s],t)
        np.add.at(r2, r, messages)
        r2 *= self.alignment / n

        #Separation
        r3 = np.zeros_like(pos)
        adj2 = np.argwhere(d2 < self.avoidance_range**2)
        r,s = adj2[:,0], adj2[:,1]
        u,n = np.unique(r, return_counts=True)
        n = np.maximum(n[:,np.newaxis]-1,1)
        messages = self.message_passing(pos[r], pos[s], t)
        norms = np.linalg.norm(messages, ord=2, axis=1, keepdims=True)
        norms[norms==0] = 1
        messages = messages / norms
        np.subtract.at(r3, r, messages)
        r3 *= self.separation / n
        
        #Bounding Box
        box = np.array([[0,0],[10,10]])
        r4 = np.zeros_like(pos)
        idx0 = pos[:,0] < (box[0,0] + self.avoidance_range)
        r4[idx0,0] = 0.5
        r4[idx0,1] = np.sign(vel[idx0,1]) * 0.5
        idx1 = pos[:,0] > box[1,0] - self.avoidance_range
        r4[idx1,0] = -0.5
        r4[idx1,1] = np.sign(vel[idx1,1]) * 0.5
        
        idx2 = pos[:,1] < box[0,1] + self.avoidance_range
        idxx = ~ (idx0 | idx1)
        r4[idx2 * idxx, 0] = np.sign(vel[idx2*idxx,1]) * 0. #but
        r4[idx2, 1] = 0.5
        idx3 = pos[:,1] > box[1,1] - self.avoidance_range
        r4[idx3*idxx,0] = np.sign(vel[idx3*idxx,1]) * 0.5 #but
        r4[idx3,1] = -0.5

        #Limit Acceleration
        acc = r1 + r2 + r3 + r4
        acc = self.normalize_max(acc, self.max_acc)

        return np.hstack([vel, acc])        
    
systems = {
    "RCNet" : RC_Network,
    "Gravitational": Gravitational_System,
    "Boids": Boids
}