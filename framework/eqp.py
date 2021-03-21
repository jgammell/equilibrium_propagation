import numpy as np
import torch
import time
from framework import datasets

def ttos(t_ns):
    if t_ns < 1e3:
        return '%.03f nsec'%(t_ns)
    elif 1e3 <= t_ns < 1e6:
        return '%.03f usec'%(t_ns / 1e3)
    elif 1e6 <= t_ns < 1e9:
        return '%.03f msec'%(t_ns / 1e6)
    elif 1e9 <= t_ns < 60e9:
        return '%.03f sec'%(t_ns / 1e9)
    elif 60e9 <= t_ns < 3600e9:
        return '%.03f min'%(t_ns / 60e9)
    elif 3600e9 <= t_ns:
        return '%.03f hr'%(t_ns / 3600e9)

def rms(A):
    return torch.sqrt(torch.norm(A, 'fro')/torch.numel(A))

def rho(s):
    return torch.clamp(s, 0, 1)

def rhoprime(s, device):
    rp = torch.zeros(s.shape).to(device)
    rp[(0<=s) & (s<=1)] = 1
    return rp

class Network:
    def __init__(self, topology, hyperparameters, configuration, dataset):
        
        np.set_printoptions(precision=2, linewidth=320)
        torch.set_printoptions(precision=2, linewidth=320)

        print('Parsing network settings...')
        t0 = 1e9*time.time()
        assert type(topology['layer sizes'] == list)
        for ls in topology['layer sizes']:
            assert type(ls) == int
            assert ls > 0
        self.layer_sizes = topology['layer sizes']
        assert topology['network type'] in ['MLFF', 'SW_intra', 'SW_no_intra']
        self.network_type = topology['network type']
        if self.network_type in ['SW_intra', 'SW_no_intra']:
            assert type(topology['bypass p']) == float
            assert 0 <= topology['bypass p'] <= 1
            assert type(topology['bypass mag']) == float
            assert topology['bypass mag'] >= 0
            self.bypass_p = topology['bypass p']
            self.bypass_mag = topology['bypass mag']
        elif self.network_type in ['MLFF']:
            self.bypass_p = None
            self.bypass_mag = None
        else:
            assert False
        if type(hyperparameters['learning rate']) == float:
            assert hyperparameters['learning rate'] > 0
            self.learning_rate = hyperparameters['learning rate']
        elif type(hyperparameters['learning rate']) == list:
            for lr in hyperparameters['learning rate']:
                assert type(lr) == float
                assert lr > 0
            assert len(hyperparameters['learning rate']) == (len(self.layer_sizes)-1)
            self.learning_rate = hyperparameters['learning rate']
        else:
            assert False
        assert type(hyperparameters['epsilon']) == float
        assert hyperparameters['epsilon'] > 0
        self.epsilon = hyperparameters['epsilon']
        assert type(hyperparameters['beta']) == float
        assert hyperparameters['beta'] > 0
        self.beta = hyperparameters['beta']
        assert type(hyperparameters['free iterations']) == int
        assert hyperparameters['free iterations'] > 0
        self.free_iterations = hyperparameters['free iterations']
        assert type(hyperparameters['weakly clamped iterations']) == int
        assert hyperparameters['weakly clamped iterations'] > 0
        self.weakly_clamped_iterations = hyperparameters['weakly clamped iterations']
        assert type(configuration['batch size']) == int
        assert configuration['batch size'] > 0
        self.batch_size = configuration['batch size']
        assert configuration['device'] in ['cpu', 'CUDA:0', 'cuda']
        self.device = configuration['device']
        assert dataset in [datasets.MNIST, datasets.FashionMNIST, datasets.Diabetes, datasets.Wine]
        assert type(configuration['seed']) == int
        assert configuration['seed'] >= 0
        self.seed = configuration['seed']
        print('\tCompleted successfully')
        print('\tTime taken: %s'%(ttos((1e9*time.time())-t0)))
        print('Initializing network...')
        t0 = 1e9*time.time()
        print('\tInitializing indices...')
        t1 = 1e9*time.time()
        self.layer_indices = np.cumsum([0] + self.layer_sizes)
        self.num_neurons = np.sum(self.layer_sizes)
        self.ix = slice(0, self.layer_indices[1])
        self.ih = slice(self.layer_indices[1], self.layer_indices[-2])
        self.iy = slice(self.layer_indices[-2], self.layer_indices[-1])
        self.ihy = slice(self.layer_indices[1], self.layer_indices[-1])
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing seeds...')
        t1 = 1e9*time.time()
        torch.manual_seed(seed=self.seed)
        np.random.seed(seed=self.seed)
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing dataset...')
        t1 = 1e9*time.time()
        self.dataset = dataset(self.batch_size, self.device)
        print('\t\tCompleted successfully.')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing state...')
        t1 = 1e9*time.time()
        self.initialize_state()
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing persistent particles...')
        t1 = 1e9*time.time()
        self.initialize_persistent_particles()
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing weights...')
        t1 = 1e9*time.time()
        self.initialize_weight_matrices()
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tInitializing biases...')
        t1 = 1e9*time.time()
        self.initialize_biases()
        print('\t\tCompleted successfully')
        print('\t\tTime taken: %s'%(ttos((1e9*time.time())-t1)))
        print('\tCompleted successfully.')
        print('\tTime taken: %s'%(ttos((1e9*time.time())-t0)))
        print('Network initialized successfully.')
        print('\tLayer sizes: %s'%('-'.join([str(val) for val in self.layer_sizes])))
        print('\tNetwork type: %s'%(self.network_type))
        print('\tBypass p: ' + ('n/a' if (self.bypass_p == None) else '%f'%(self.bypass_p)))
        print('\tBypass magnitude: ' + ('n/a' if (self.bypass_mag == None) else '%f'%(self.bypass_mag)))
        print('\tLearning rate:', self.learning_rate)
        if type(self.learning_rate) == list:
            print('\t\tUsing per-layer rates.')
        elif type(self.learning_rate) == float:
            print('\t\tUsing a single global learning rate.')
        else:
            assert False
        print('\tEpsilon: %f'%(self.epsilon))
        print('\tBeta: %f'%(self.beta))
        print('\tFree iterations: %d'%(self.free_iterations))
        print('\tWeakly-clamped iterations: %d'%(self.weakly_clamped_iterations))
        print('\tDataset: %s'%(self.dataset.name))
        print('\t\tInput: %d'%(self.dataset.n_in))
        print('\t\tOutput: %d'%(self.dataset.n_out))
        print('\t\tTraining batches: %d'%(self.dataset.n_trainb))
        print('\t\tTest batches: %d'%(self.dataset.n_testb))
        print('\t\tBatch size: %d'%(self.dataset.batch_size))
        print('\t\tClassification: %r'%(self.dataset.classification))
        print('\tBatch size: %d'%(self.batch_size))
        print('\tDevice: %s'%(self.device))
        print('\tSeed: %d'%(self.seed))
        print('\tState:')
        print('\t\tRMS value: %f'%(rms(self.s)))
        print('\t\tShape: '+' x '.join([str(val) for val in list(self.s.shape)]))
        print('\tPersistent particles:')
        print('\t\tNumber of persistent particles: %d'%(len(self.persistent_particles)))
        print('\t\tMax RMS persistent particle: %f'%(np.max([rms(pp) for pp in self.persistent_particles])))
        for pp in self.persistent_particles:
            assert pp.shape == self.persistent_particles[0].shape
        print('\t\tShape: ' + ' x '.join([str(val) for val in self.persistent_particles[0].shape]))
        print('\tWeight matrices:')
        print('\t\tActual p: %.03f'%(self.p_actual))
        print('\t\tRMS W element: %f'%(rms(self.W)))
        print('\t\tRMS W_mask element: %f'%(rms(self.W_mask)))
        print('\t\tW shape: ' + ' x '.join([str(val) for val in self.W.shape]))
        print('\t\tW_mask shape: ' + ' x '.join([str(val) for val in self.W_mask.shape]))
        for conn in self.interlayer_connections:
            assert conn.shape == self.interlayer_connections[0].shape
        print('\t\tInterlayer connection mask shape: ' + ' x '.join([str(val) for val in self.interlayer_connections[0].shape]))

    def initialize_state(self):
        self.s = torch.zeros(self.batch_size, self.num_neurons, dtype=torch.float32).to(self.device)

    def initialize_persistent_particles(self):
        self.persistent_particles = [
            torch.zeros(self.s[:, self.ihy].shape, dtype=torch.float32).to(self.device)
            for _ in range(self.dataset.n_trainb + self.dataset.n_testb)]
    
    def use_persistent_particle(self, index):
        assert 0 <= index < len(self.persistent_particles)
        self.s[:, self.ihy] = self.persistent_particles[index].clone()
    
    def update_persistent_particle(self, index):
        assert 0 <= index < len(self.persistent_particles)
        self.persistent_particles[index] = self.s[:, self.ihy].clone()

    def initialize_weight_matrices(self):
        self.rng = np.random.RandomState(seed=self.seed)
        W = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        W_mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        interlayer_connections = []
        for i, j, k in zip(self.layer_indices[:-2], self.layer_indices[1:-1], self.layer_indices[2:]):
            conn = np.zeros(W.shape, dtype=np.int32)
            conn[j:k, i:j] = 1
            conn[i:j, j:k] = 1
            interlayer_connections.append(conn)
        for conn in interlayer_connections:
            W_mask += conn
        potential_conn_indices = []
        for i in range(2, len(self.layer_indices)-1):
            for row in range(self.layer_indices[i], self.layer_indices[-1]):
                for col in range(self.layer_indices[i-2], self.layer_indices[i-1]):
                    assert W_mask[row, col] == 0
                    potential_conn_indices.append([row, col])
        self.p_actual = 0
        initial_length = 1
        if self.network_type == 'MLFF':
            pass
        elif self.network_type == 'SW_intra':
            for i, j in zip(self.layer_indices[1:-2], self.layer_indices[2:-1]):
                assert np.linalg.norm(W_mask[i:j, i:j], ord='fro') == 0
                W_mask[i:j, i:j] = 1
            existing_conn_indices = []
            for col in range(W_mask.shape[0]):
                for row in range(col, W_mask.shape[0]):
                    if W_mask[row, col] != 0:
                        assert W_mask[row, col] == 1
                        existing_conn_indices.append([row, col])
            initial_length = len(existing_conn_indices)
            for idx, conn in enumerate(existing_conn_indices):
                if self.rng.uniform(0, 1) < self.bypass_p:
                    new_conn_idx = self.rng.randint(len(potential_conn_indices))
                    new_conn = potential_conn_indices[new_conn_idx]
                    W_mask[conn[0], conn[1]] = 0
                    W_mask[conn[1], conn[0]] = 0
                    W_mask[new_conn[0], new_conn[1]] = 1
                    W_mask[new_conn[1], new_conn[0]] = 1
                    potential_conn_indices.append(conn)
                    del potential_conn_indices[new_conn_idx]
                    self.p_actual += 1
#            for i in range(int(self.bypass_p*len(existing_conn_indices))):
#                assert len(potential_conn_indices) > 0
#                existing_location_index = self.rng.randint(len(existing_conn_indices))
#                existing_conn = existing_conn_indices[existing_location_index]
#                new_location_index = self.rng.randint(len(potential_conn_indices))
#                new_conn = potential_conn_indices[new_location_index]
#                W_mask[existing_conn[0], existing_conn[1]] = 0
#                W_mask[existing_conn[1], existing_conn[0]] = 0
#                W_mask[new_conn[0], new_conn[1]] = 1
#                W_mask[new_conn[1], new_conn[0]] = 1
#                del existing_conn_indices[existing_location_index]
#                del potential_conn_indices[new_location_index]
#                self.p_actual += 1
            W += np.asarray(self.rng.uniform(low=-self.bypass_mag, high=self.bypass_mag, size=W.shape))
        elif self.network_type == 'SW_no_intra':
            existing_conn_indices = []
            for row in range(1, W_mask.shape[0]):
                for col in range(row):
                    if W_mask[row, col] != 0:
                        assert W_mask[row, col] == 1
                        existing_conn_indices.append([row, col])
            initial_length = len(existing_conn_indices)
            for idx, conn in enumerate(existing_conn_indices):
                if self.rng.uniform(0, 1) < self.bypass_p:
                    new_conn_idx = self.rng.randint(len(potential_conn_indices))
                    new_conn = potential_conn_indices[new_conn_idx]
                    W_mask[conn[0], conn[1]] = 0
                    W_mask[conn[1], conn[0]] = 0
                    W_mask[new_conn[0], new_conn[1]] = 1
                    W_mask[new_conn[1], new_conn[0]] = 1
                    potential_conn_indices.append(conn)
                    del potential_conn_indices[new_conn_idx]
                    self.p_actual += 1
            
#            for i in range(int(self.bypass_p*len(existing_conn_indices))):
#                assert len(potential_conn_indices) > 0
#                existing_location_index = self.rng.randint(len(existing_conn_indices))
#                existing_conn = existing_conn_indices[existing_location_index]
#                new_location_index = self.rng.randint(len(potential_conn_indices))
#                new_conn = potential_conn_indices[new_location_index]
#                W_mask[existing_conn[0], existing_conn[1]] = 0
#                W_mask[existing_conn[1], existing_conn[0]] = 0
#                W_mask[new_conn[0], new_conn[1]] = 1
#                W_mask[new_conn[1], new_conn[0]] = 1
#                del existing_conn_indices[existing_location_index]
#                del potential_conn_indices[new_location_index]
#                self.p_actual += 1
            W += np.asarray(self.rng.uniform(low=-self.bypass_mag, high=self.bypass_mag, size=W.shape))
        else:
            assert False
        self.p_actual /= initial_length
        for conn, n_in, n_out in zip(interlayer_connections, self.layer_sizes[:-1], self.layer_sizes[1:]):
            W -= W*conn
            W += conn*np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in+n_out)),
                    high = np.sqrt(6. / (n_in+n_out)),
                    size=W.shape))
        W *= W_mask
        W = np.tril(W, k=-1) + np.tril(W, k=-1).T
        W_mask = np.tril(W_mask, k=-1) + np.tril(W_mask, k=-1).T
        self.W = torch.from_numpy(W).float().to(self.device).unsqueeze(0)
        self.W_mask = torch.from_numpy(W_mask).float().to(self.device).unsqueeze(0)
        self.interlayer_connections = [torch.from_numpy(conn).float().to(self.device).unsqueeze(0) for conn in interlayer_connections]
        assert (self.W - (self.W*self.W_mask)).norm() == 0
        assert (self.W - (self.W.tril() + self.W.tril().transpose(1, 2))).norm() == 0
        assert self.W.norm() != 0
        
    def initialize_biases(self):
        self.B = torch.zeros(self.s.shape).to(self.device)
        
    def set_x_state(self, x):
        assert x.shape == self.s[:, self.ix].shape
        self.s[:, self.ix] = x
    
    def set_y_state(self, y):
        assert y.shape == self.s[:, self.iy].shape
        self.s[:, self.iy] = y
    
    def calc_E(self):
        term1 = .5*torch.sum(self.s*self.s, dim=1)
        rho_s = rho(self.s)
        term2 = torch.matmul(rho_s.unsqueeze(2), rho_s.unsqueeze(1))
        term2 *= self.W
        term2 = -.5*torch.sum(term2, dim=[1, 2])
        term3 = -np.sum([self.B[i]*rho(s[i]) for i in range(len(b))])
        return term1 + term2 + term3
    
    def calc_C(self, y_target):
        y = self.s[:, self.iy]
        return .5*torch.norm(y-y_target, dim=1)**2
    
    def calc_F(self, y_target):
        return self.calc_E() + self.beta*self.calc_C(y_target)
        
    def step_free(self, y):
        Rs = (rho(self.s)@self.W).squeeze()
        dEds = self.epsilon*(Rs+self.B-rho(self.s))
        dEds[:, self.ix] = 0
        self.s += dEds
        torch.clamp(self.s, 0, 1, out=self.s)
    
    def step_weakly_clamped(self, y):
        Rs = (rho(self.s)@self.W).squeeze()
        dEds = self.epsilon*(Rs+self.B-rho(self.s))
        dEds[:, self.ix] = 0
        self.s += dEds
        dCds = self.epsilon*self.beta*(y-self.s[:, self.iy])
        self.s[:, self.iy] += 2*dCds
        torch.clamp(self.s, 0, 1, out=self.s)
        
    def evolve_to_equilibrium(self, phase, y=None):
        if phase == 'free':
            iterations = self.free_iterations
            step = self.step_free
        elif phase == 'weakly-clamped':
            iterations = self.weakly_clamped_iterations
            step = self.step_weakly_clamped
        else:
            assert False
        for iteration in np.arange(iterations):
            step(y)
    
    def calculate_weight_update(self, s_free_phase, s_clamped_phase):
        term1 = torch.unsqueeze(rho(s_clamped_phase), dim=2)@torch.unsqueeze(rho(s_clamped_phase), dim=1)
        term2 = torch.unsqueeze(rho(s_free_phase), dim=2)@torch.unsqueeze(rho(s_free_phase), dim=1)
        dW = (1/self.beta)*(term1-term2)
        dW *= self.W_mask
        self.dW = torch.mean(dW, dim=0)#.unsqueeze(0)
    
    def calculate_bias_update(self, s_free_phase, s_clamped_phase):
        dB = (1/self.beta)*(rho(s_clamped_phase)-rho(s_free_phase))
        dB[:, self.ix] = 0
        self.dB = torch.mean(dB, dim=0)#.unsqueeze(0)
    
    def train_batch(self, x, y, index, classification=False):
        self.use_persistent_particle(index)
        self.set_x_state(x)
        self.evolve_to_equilibrium('free')
        self.s_free_phase = self.s.clone()
        self.update_persistent_particle(index)
        n_correct = None
        if classification:
            n_correct = int(torch.eq(torch.argmax(self.s[:, self.iy], dim=1), torch.argmax(y, dim=1)).sum().cpu())
        cost = float(torch.mean(self.calc_C(y)).cpu())
        assert torch.norm(self.s[:, self.ix]-x) <= 1e-5
        assert torch.norm(self.W - (torch.tril(self.W, diagonal=-1) + torch.tril(self.W, diagonal=-1).transpose(1, 2))) == 0
        assert torch.norm(self.W - (self.W*self.W_mask)) == 0
        if np.random.randint(0, 2):
            self.beta *= -1
        self.evolve_to_equilibrium('weakly-clamped', y)
        self.s_clamped_phase = self.s.clone()
        self.calculate_weight_update(self.s_free_phase, self.s_clamped_phase)
        self.calculate_bias_update(self.s_free_phase, self.s_clamped_phase)
        if type(self.learning_rate) == float:
            self.W += self.learning_rate*self.dW
            self.B += self.learning_rate*self.dB
        elif type(self.learning_rate) == list:
            dWlr = self.dW.clone().unsqueeze(0)
            dBlr = self.dB.clone().unsqueeze(0)
            for lr, conn in zip(self.learning_rate, self.interlayer_connections):
                dWlr[conn!=0] *= lr
            for lr, i, j in zip(self.learning_rate, self.layer_indices[1:-1], self.layer_indices[2:]):
                dBlr[i:j] *= lr
            dWlr = dWlr.tril(diagonal=-1)+dWlr.tril(diagonal=-1).transpose(1, 2)
            self.W += dWlr
            self.B += dBlr
        else:
            assert False
        return (cost, n_correct)
    
    def train_next_batch(self):
        (index, (x, y)) = self.dataset.next_training_batch()
        (cost, n_correct) = self.train_batch(x, y, index, self.dataset.classification)
        return n_correct if self.dataset.classification else cost
        
    def train_epoch(self):
        self.training_error = 0
        self.mean_dW = torch.zeros(self.W.shape).to(self.device)
        self.per_layer_rates = []
        for batch in range(self.dataset.n_trainb):
            error = self.train_next_batch()
            if self.dataset.classification:
                self.training_error += (self.batch_size-error)
            else:
                self.training_error += error
            self.per_layer_rates.append([])
            for conn in self.interlayer_connections:
                correction = torch.norm((self.dW*conn)/torch.sqrt(torch.norm(self.W_mask*conn, p=1)))
                self.per_layer_rates[-1].append(float(correction.cpu()))
            self.mean_dW = torch.abs(self.dW)/(batch+1) + (batch/(batch+1))*self.mean_dW
            #print('\tBatch %d complete.'%(batch+1))
        if self.dataset.classification:
            self.training_error /= (self.dataset.n_trainb*self.batch_size)
        else:
            self.training_error /= self.dataset.n_trainb

    def calculate_training_error(self):
        self.true_training_error = 0
        for batch in range(self.dataset.n_trainb):
            (index, (x, y)) = self.dataset.next_training_batch()
            self.set_x_state(x)
            self.use_persistent_particle(index)
            self.evolve_to_equilibrium('free')
            #self.update_persistent_particle(index)
            if self.dataset.classification:
                self.true_training_error += int(torch.eq(torch.argmax(self.s[:, self.iy], dim=1), torch.argmax(y, dim=1)).sum())
            else:
                self.true_training_error += float(torch.mean(self.calc_C(y)))
        if self.dataset.classification:
            self.true_training_error = 1 - (self.true_training_error/(self.dataset.n_trainb*self.batch_size))
        else:
            self.true_training_error /= (self.dataset.n_trainb)#*self.batch_size)

    def calculate_test_error(self):
        self.test_error = 0
        for batch in range(self.dataset.n_testb):
            (index, (x, y)) = self.dataset.next_test_batch()
            self.set_x_state(x)
            self.use_persistent_particle(index)
            self.evolve_to_equilibrium('free')
            self.update_persistent_particle(index)
            if self.dataset.classification:
                self.test_error += int(torch.eq(torch.argmax(self.s[:, self.iy], dim=1), torch.argmax(y, dim=1)).sum())
            else:
                self.test_error += float(torch.mean(self.calc_C(y)))
        if self.dataset.classification:
            self.test_error = 1 - (self.test_error/(self.dataset.n_testb*self.batch_size))
        else:
            self.test_error /= (self.dataset.n_testb)#*self.batch_size)
        
        
        
        
        
        
