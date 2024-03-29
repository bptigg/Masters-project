import numpy as np
from scipy import linalg
from scipy import integrate
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import qutip as qt
import multiprocessing
import datetime
import select


operator_function_mapping = {
    'x' : lambda dim: qt.spin_Jx((dim-1)/2),
    'y' : lambda dim: qt.spin_Jy((dim-1)/2),
    'z' : lambda dim: qt.spin_Jz((dim-1)/2),
    'p' : lambda dim: qt.spin_Jp((dim-1)/2),
    'm' : lambda dim: qt.spin_Jm((dim-1)/2),
    'i' : qt.identity}

constants = {
    'g_electron' : 2.00231930436256,
    'mu_bohr'    : 9.274009994e-24, #bohr magneton
    'h_bar'      : 6.62607015e-34 / (2*np.pi),
    'mu_naught'  : (4*np.pi) * 1e-7
}

model_parameters = {
    'rfW' : np.array([  8.51061  , -14.251621 ,   6.5492562]),
    'B_0' : 1.4 * 2 * np.pi,
    'k_b' : 2., #1/us
    'k_f' : 1.,
    'angle_step' : 2*np.pi / 72,
    'run_time' : 20,  
    'absolute tolerance' : 1e-10,
    'relative tolerance' : 1e-8
}

hyperfine_prefactor = constants['g_electron'] * constants['mu_bohr'] / constants['h_bar']*1e-9

Nuclei = {
    'N5' : hyperfine_prefactor * np.array([[-0.0994933,   0.00287092, 0],
                                           [0.00287092, -0.0874862,  0.], 
                                           [0.,          0.,         1.75687]]),
    'N1' : hyperfine_prefactor * np.array([[-0.0529598, -0.0586562, 0.0460172], 
                                           [-0.0586562,  0.564443, -0.564764],
                                           [0.0460172, -0.564764,  0.453074]])
    
}




class structure:
    nuclear_spins = []
    hypefine_matricies = []
    radical_seperation = np.array([0,0,0])

    hypefine_list_length = 0
    
    def __init__(self, spins, matricies, seperation) -> None:
        self.hypefine_matricies = matricies
        self.nuclear_spins = spins
        self.radical_seperation = seperation
        self.hypefine_list_length = len(matricies)

    def get_electron_sep(self):
        return self.radical_seperation

    def get_nuclear_spins(self):
        return self.nuclear_spins
    
    def get_hyperfine_matracies(self, i = -1, length = False):
        if(length == True):
            return self.hypefine_list_length
        if(i != -1):
            return self.hypefine_matricies[i]
        else:
            return self.hypefine_matricies
        

def make_spin_operator(dims, specs):
    ops = [qt.identity(d) for d in dims]
    for ind, opstr in specs:
        ops[ind] = ops[ind] * operator_function_mapping[opstr](dims[ind])
    return qt.tensor(ops)

def identiy_operator(dims):
    return make_spin_operator(dims, [])

def zero_operator(dims):
    d = np.prod(dims)
    return qt.Qobj(sparse.csr_matrix((d,d), dtype=np.float64),dims=[list(dims)]*2, type="oper", isherm=True)

def make_Hamiltionian(dims, ind, parvec): #single particle interaction
    axes = ['x', 'y', 'z']
    components = [v * make_spin_operator(dims, [(ind,ax)]) for  v, ax in zip(parvec, axes) if v != 0]
    if components:
        return sum(components)
    else:
        return zero_operator(dims)
    
def make_Hamiltonian_2(dims, ind_1, ind_2, parmat):
    axes = ['x', 'y', 'z']
    components = []
    for i in range(3):
       for j in range(3):
           if parmat[i,j] != 0:
               components.append(parmat[i,j] * make_spin_operator(dims, [(ind_1,axes[i]), (ind_2,axes[j])]))
    if components:
        return sum(components)
    else:
        return zero_operator(dims)

def point_dipole_dipole_coupling(r):
    C = -1* (constants['mu_naught'] / (4*np.pi*1e-30)) * (constants['g_electron'] * constants['mu_bohr'])**2 * 1/(1e6*constants['h_bar']*2*np.pi)
    
    if np.isscalar(r):
        d = C / r**3
        A = np.diag([-d, -d, -2*d])
    else:
        r_norm = np.linalg.norm(r)
        d = C / r_norm**3
        e = r / r_norm
        A = d * (3 * e[:,np.newaxis] * e[np.newaxis,:] - np.eye(3))
    
    return A 

def polar_to_cart(theta : float, phi : float): #r = 1, theta = inclanation, phi = azimuthal 
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x,y,z])


def radical_pair_model(model : structure, processes : int):
    DDSE = point_dipole_dipole_coupling(model.get_electron_sep()) * (2*np.pi)
    #DDSE = 0 * DDSE #dipole dipole spin exchange

    dims = [2,2, *[round(2*I+1) for I in model.get_nuclear_spins()]]

    initial_projection_operator = 1/4 * make_spin_operator(dims, []) - make_Hamiltonian_2(dims, 0, 1, np.identity(3)) 
    identity = make_spin_operator(dims, [])
    triplet_projection_operator = identity - initial_projection_operator

    Hyperfine_hamiltonian = sum(make_Hamiltonian_2(dims, i, i+2, model.get_hyperfine_matracies(i)) for i in range(model.get_hyperfine_matracies(length=True)))
    dipole_dipole_electron_hamiltonian = make_Hamiltonian_2(dims, 0, 1, DDSE)

    K = model_parameters['k_b']/2 * initial_projection_operator + model_parameters['k_f']/2 * identity
    rho_0 = initial_projection_operator / initial_projection_operator.tr()
    H_0 = Hyperfine_hamiltonian + dipole_dipole_electron_hamiltonian

    steps = np.pi / model_parameters['angle_step']
    steps_2 = int(np.ceil(steps))

    new = False

    if(steps_2 == steps):
        orientation = [*[np.arange(0 , 2* np.pi + model_parameters['angle_step']  , model_parameters['angle_step'],) for i in range(0,steps_2+1)]]
    else:
        new_step = (np.pi / steps_2)
        new = True
        orientation = [*[np.arange(0, 2* np.pi + new_step, new_step) for i in range(0,steps_2+1)]]

    t_max = model_parameters['run_time'] / model_parameters['k_f']
    #t_list = np.linspace(0,t_max, int(np.ceil(1000*t_max)))

    #tolerance options
    global opt
    opt = qt.Options()
    opt.atol = model_parameters['absolute tolerance']
    opt.rtol = model_parameters['relative tolerance']

    step = 0
    global operation_number
    operation_number = 0
    operation_lock = multiprocessing.Lock()

    global len_theta
    global operations

    if new:
        step = new_step
    else:
        step = model_parameters['angle_step']

    if(processes > len(orientation)):
        processes = len(orientation)

    orientations = []
    size = len(orientation) / processes
    size = int(np.ceil(size))
    processes = int(len(orientation) / size)
    for i in range(0,processes):
        orientations.append(orientation[i*size : (i+1)*size])

    number_of_cores = multiprocessing.cpu_count()
    cores_per_process = float(number_of_cores) / float(processes)
    cores_per_process = int(np.floor(cores_per_process))
    opt.num_cpus = cores_per_process
    opt.openmp_threads = cores_per_process

    len_theta  = len(orientation[0])
    operations = (len(orientation) * len_theta) / processes


    process_id = []
    pipe = []
    for i in range(0,processes):
        if __name__ == '__main__':
            reciever, sender = multiprocessing.Pipe(True)
            reciever_2, sender_2 = multiprocessing.Pipe(True)
            p = multiprocessing.Process(target = simulation, args = (dims, H_0, orientations[i], step, rho_0, initial_projection_operator, t_max, K, operation_lock, i * size, [sender, reciever_2]))
            process_id.append(p)
            pipe.append([reciever, sender_2])

    none_list = []
    for p in process_id:
        p.start()
        none_list.append(False)

    if __name__ == '__main__':

        results = []

        while(len(process_id) != 0):
            remove = []
            for i in range(0,len(process_id)):
                joined = False

                #if(pipe[i][1].poll(0.01)):
                #    if(pipe[i][1].recv() == None):
                #        print("Process {} ended".format(process_id[i]._identity))
                #
                #elif(pipe[i][0].poll(0.01) == True):
                #    results.append(pipe[i][0].recv())
                #    if(pipe[i][1].closed == False):
                #        if(process_id[i].is_alive() == True):
                #            pipe[i][1].send(True)

                #process_id[i].join(0.01)
                #if(process_id[i].is_alive() == False):
                #    remove.append(True)
                #    joined = True
                #else:
                #    remove.append(False)

                process_id[i].join(0.01)
                if(process_id[i].is_alive() == False):
                    remove.append(True)
                else:
                    remove.append(False)
                
                if(none_list[i] != True):
                    if(pipe[i][0].poll(0.01) == True):
                        result = pipe[i][0].recv()
                        if(result == None):
                            none_list[i] = True
                        else:
                            results.append(result)

                
                
                #if(joined != True and pipe[i][0].closed == False):
                #    if(pipe[i][0].poll(0.1) == True):
                #        results.append(pipe[i][0].recv())
                #        if(pipe[i][1].closed == False):
                #            pipe[i][1].send(True)
                

            removed = 0
            for i in range(0, len(remove)):
                if remove[i]:
                    del(process_id[i-removed])
                    del(pipe[i-removed])
                    removed = removed+1

        #results = [x.recv() for x in pipe]

        data = [[],[],[],[]]
        for i in results:
            list = 0
            for e in i:
                for f in range(0,len(e)):
                    data[list].append(e[f])
                list = list + 1

        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
        f = open(dt_string + ".txt", "w")

        for i in range(0,len(data[0])):
            line = str(data[0][i]) + " " + str(data[1][i]) + " " + str(data[2][i]) + " " + str(data[3][i]) + "\n"
            f.write(line)
        f.close()

def simulation(dims, H_0, orientation, step, rho_0, Ps, t_max, K, operation_lock : multiprocessing.Lock , start_index, return_data):
    global operation_number
    t_list = np.linspace(0,t_max, int(np.ceil(1000*t_max)))
    
    x = []
    y = []
    z = []
    yields = []
    index = start_index
    send_data_permission = True

    for phi in orientation:
        for theta in phi:
            ori = polar_to_cart(theta, index * step)

            x.append(ori[0])
            y.append(ori[1])
            z.append(ori[2])

            B_field = model_parameters['B_0'] * ori

            H_zee = make_Hamiltionian(dims, 0, B_field) + make_Hamiltionian(dims, 1, B_field)
            H_eff = H_0 + H_zee - 1j*K

            #L_eff = -1j*qt.spre(H_eff) + 1j*qt.spost(H_eff.conj().trans())
            #sol = qt.mesolve(L_eff, rho_0, t_list, e_ops=[Ps], options=opt)
            #ps = sol.expect[0]
            #yr = model_parameters['k_b'] * integrate.simps(ps, t_list)
            #yields.append(yr)
            yields.append(ori[2]+ori[1]+ori[0])
            operation_lock.acquire()
            operation_number = operation_number + 1
            
            while(send_data_permission != True):
                if(return_data[0].poll(0.1) == True):
                    send_data_permission = return_data[0].recv()
                
            return_data[0].send([x,y,z,yields])
            x.clear()
            y.clear()
            z.clear()
            yields.clear()

            #if(operation_number % 10 == 0):
            #    #have a pipe that sends a boolean back from the main thread to tell the sub threads that it's okay to send data back up the pipe    
            #    if(send_data_permission == False):
            #        if(return_data[1].poll(0.1) == True):
            #            permission = [return_data[1].recv()]      
            #            send_data_permission = permission[0]
            #    return_data[0].send([x,y,z,yields])
            #    send_data_permission = False
            #    x.clear()
            #    y.clear()
            #    z.clear()
            #    yields.clear()

            print("Operation: {}/{}".format(operation_number, operations))
            operation_lock.release()
        index = index + 1  
    
    #send_data_permission = False
    #while(send_data_permission == False):
    #    if(return_data[1].poll() == True):
    #        permission = [return_data[1].recv()]      
    #        send_data_permission = permission[0]

    return_data[0].send(None)


def main():
    radical_pair_model(structure([1,1], [Nuclei['N5'], Nuclei['N1']], model_parameters['rfW']), 6)



main()


