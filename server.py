import torch
import numpy as np
import itertools
#FedAvg
def average_models(model_params_list):
    averaged_params = {}

    for key in model_params_list[0].keys():
        averaged_params[key] = sum([client[key] for client in model_params_list]) / len(model_params_list)

    return averaged_params


def average_gradients(grads_list,M_t):#, client_lengths):
    agg = {}
    for key in grads_list[0]:
        total = 0
        for i,g in enumerate(grads_list) : 
            #print(f"length of dataset for client {i}: {client_lengths[i]}")
            #print(f"shape of gradient for client {i} : {g[key].shape}")
            total += (g[key])#* (client_lengths[i]/sum(client_lengths))
        agg[key] = total / M_t #/ len(gra ds_list)
    return agg

"""
def average_gradients(grads_list,M_t):#, client_lengths):
    agg = {}
    keys = grads_list[0].keys()
    for key in keys :
        agg[key] = sum(g[key] for g in grads_list) / M_t 
    return agg
"""

def real_expectation(m,grads_list,epsilon):
    M = len(grads_list)
    sums = np.zeros(M)
    clients = list(range(M))
    for k in range(1,M+1):
        subsets = list(itertools.combinations([client for client in clients if client != m], k-1))
        sum_i_j = 0
        #print("subsets : " , subsets)
        for C in subsets : 
            prod_i = np.prod([1 - epsilon[client] for client in C]) 
            prod_j = np.prod([epsilon[client] for client in clients if  client != m  and client not in C ])
            sum_i_j += (prod_i * prod_j )
    

        sums[k-1] = ((1-epsilon[m])/k) * sum_i_j
        #print(f"iteration k : {k} sum : {sums[k-1]}")
        
    return np.sum(sums)

def all_combinations_up_to_length(elements, max_length):
    return list(itertools.chain.from_iterable(
        itertools.combinations(elements, r) for r in range(1, max_length + 1)
    ))

def real_expectation_corr(m,M, eps_active, eps_nonactive ,epsilons_access,ap_list,group1,group2):
    AP_subsets = all_combinations_up_to_length(ap_list , len(ap_list)+1)
    clients = np.array(list(range(M)))
    eps_active = np.array([eps_active[key] for key in eps_active.keys()])
    eps_nonactive = np.array([eps_nonactive[key] for key in eps_nonactive.keys()])

    #eps_nonactive = np.array(eps_nonactive)
    sum_ap = 0 
    for M_ap in AP_subsets :
        sums = np.zeros(M)
        group_mask = np.isin(clients, group1)  # True where in group1
        aps = np.where(group_mask, 1, 2)  # AP assignment: 1 or 2
        ap_in_subset_mask = np.isin(aps, M_ap)
        epsilons = np.where(ap_in_subset_mask, eps_active[clients], eps_nonactive[clients])

        for k in range(1,M+1):
            subsets = list(itertools.combinations([client for client in clients if client != m], k-1))
            sum_i_j = 0
            for C in subsets :
                prod_i = np.prod([1 - epsilons[client] for client in C])
                prod_j = np.prod([epsilons[client] for client in clients if client != m and client not in C])
                sum_i_j += (prod_i * prod_j )
            sums[k-1] = ((1-epsilons[m])/k) * sum_i_j
        conditional_expectation = np.sum(sums)
        prod_ap1 = np.prod([1 - epsilons_access[ap-1] for ap in M_ap]) 
        prod_ap2 = np.prod([epsilons_access[ap-1] for ap in ap_list if  ap not in M_ap])
        sum_ap += conditional_expectation * prod_ap1 * prod_ap2
    return sum_ap



def real_variance(m,grads_list,epsilon):
    M = len(grads_list)
    sums = np.zeros(M)
    clients = list(range(M))
    for k in range(1,M+1):
        subsets = list(itertools.combinations([client for client in clients if client != m], k-1))
        sum_i_j = 0
        #print("subsets : " , subsets)
        for C in subsets : 
            prod_i = np.prod([1 - epsilon[client] for client in C]) 
            prod_j = np.prod([epsilon[client] for client in clients if  client != m  and client not in C ])
            sum_i_j += (prod_i * prod_j )
    

        sums[k-1] = ((1-epsilon[m])/k**2) * sum_i_j
    return np.sum(sums) - real_expectation(m,grads_list,epsilon)**2

def monte_carlo_expectation(m,epsilon = np.ones(1),N=10000):
    estimates = np.zeros(N)
    for n in range(N):
        X = np.random.binomial(1, 1 - epsilon)  # X_k is 1 with probability 1 - epsilon_k
      
        # Calculate the set size |M_t| (number of ones in X)
        M_t_size = np.sum(X)
        
        if M_t_size > 0:
            if X[m] == 1:
                
                estimates[n] = 1 / M_t_size  #  m is in M_t
    
    # Estimate the expectation 
    return np.mean(estimates)

def monte_carlo_expectation_corr(m, group1, group2 ,  epsilons_access, eps_active, eps_nonactive, N=10000):
    estimates = []
     # Sample latent AP states
    
    for _ in range(N):
       
        X1 = np.random.binomial(1, 1 - epsilons_access[0])  # AP1
        X2 = np.random.binomial(1, 1 - epsilons_access[1])  # AP2
        #X3 = np.random.binomial(1, 1 - epsilons_access[2])  # AP3

        # Sample participation for each client
        Z = {}
        M = len(group1) + len(group2) #+ len(group3)
        for i in range(M):
            if i in group1:
                eps = eps_active[i] if X1 else eps_nonactive[i]
            elif i in group2:
                eps = eps_active[i] if X2 else eps_nonactive[i]
            """
            else :
                eps = eps_active[i] if X3 else eps_nonactive[i]
            """
            Z[i] = np.random.binomial(1, 1 - eps)

        Mt_size = sum(Z.values())
        
        if Z[m] == 1 and Mt_size > 0:
            estimates.append(1 / Mt_size)
        else:
            estimates.append(0)

    return np.mean(estimates)


def monte_carlo_expectation_3groups_corr(m, group1, group2 , group3,  epsilons_access, eps_active, eps_nonactive, N=10000):
    estimates = []
     # Sample latent AP states
    
    for _ in range(N):
       
        X1 = np.random.binomial(1, 1 - epsilons_access[0])  # AP1
        X2 = np.random.binomial(1, 1 - epsilons_access[1])  # AP2
        X3 = np.random.binomial(1, 1 - epsilons_access[2])  # AP3

        # Sample participation for each client
        Z = {}
        M = len(group1) + len(group2) + len(group3)
        for i in range(M):
            if i in group1:
                eps = eps_active[i] if X1 else eps_nonactive[i]
            elif i in group2:
                eps = eps_active[i] if X2 else eps_nonactive[i]
        
            else :
                eps = eps_active[i] if X3 else eps_nonactive[i]
            
            Z[i] = np.random.binomial(1, 1 - eps)

        Mt_size = sum(Z.values())
        
        if Z[m] == 1 and Mt_size > 0:
            estimates.append(1 / Mt_size)
        else:
            estimates.append(0)

    return np.mean(estimates)

def monte_carlo_expectation_4groups_corr(m, group1, group2 , group3, group4 , epsilons_access, eps_active, eps_nonactive, N=10000):
    estimates = []
     # Sample latent AP states
    
    for _ in range(N):
       
        X1 = np.random.binomial(1, 1 - epsilons_access[0])  # AP1
        X2 = np.random.binomial(1, 1 - epsilons_access[1])  # AP2
        X3 = np.random.binomial(1, 1 - epsilons_access[2])  # AP3
        X4 = np.random.binomial(1, 1 - epsilons_access[3]) 

        # Sample participation for each client
        Z = {}
        M = len(group1) + len(group2) + len(group3) + len(group4)
        for i in range(M):
            if i in group1:
                eps = eps_active[i] if X1 else eps_nonactive[i]
            elif i in group2:
                eps = eps_active[i] if X2 else eps_nonactive[i]
        
            elif i in group3 :
                eps = eps_active[i] if X3 else eps_nonactive[i]
            else :
                eps = eps_active[i] if X4 else eps_nonactive[i]
            Z[i] = np.random.binomial(1, 1 - eps)

        Mt_size = sum(Z.values())
        
        if Z[m] == 1 and Mt_size > 0:
            estimates.append(1 / Mt_size)
        else:
            estimates.append(0)

    return np.mean(estimates)


def monte_carlo_variance_corr(m, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000):
    estimates = []
    for _ in range(N):
        # Sample latent AP states
        X1 = np.random.binomial(1, 1 - epsilons_access[0])  # AP1
        X2 = np.random.binomial(1, 1 - epsilons_access[1])  # AP2

        # Sample participation for each client
        Z = {}
        M = len(group1) + len(group2)
        for i in range(M):
            if i in group1:
                eps = eps_active[i] if X1 else eps_nonactive[i]
            else:
                eps = eps_active[i] if X2 else eps_nonactive[i]
            Z[i] = np.random.binomial(1, 1 - eps)

        Mt_size = sum(Z.values())
        
        if Z[m] == 1 and Mt_size > 0:
            estimates.append(1 / Mt_size**2)
        else:
            estimates.append(0)

    return np.mean(estimates)


def real_expectation_gradients(grads_list , epsilon):
    agg = {}
    for key in grads_list[0]:
        total = 0
        for i,g in enumerate(grads_list):
            expectation = real_expectation(i,grads_list,epsilon)
            total += g[key] * expectation
        agg[key] = total
    return agg

def expectation_gradients(group1, group2, grads_list , epsilons_access, eps_active, eps_nonactive , M_t):
    agg = {}
    for key in grads_list[0]:
        total = 0
        for i,g in enumerate(grads_list):
            expectation = monte_carlo_expectation_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) #Monte carlo estimation of the bias
            total += g[key] * expectation
        agg[key] = total
    return agg


def real_variance_gradients(grads_list , epsilon):
    agg = {}
    for key in grads_list[0]:
        total = 0
        for i,g in enumerate(grads_list):
            variance = real_variance(i,grads_list,epsilon)
            total += g[key] * variance #Here should be the variance of the stochastic gradient estimator instead of g[key]
        agg[key] = total
    return agg

def variance_gradients(group1, group2, grads_list , epsilons_access, eps_active, eps_nonactive , M_t):
    agg = {}
    #average = average_gradients(grads_list,M_t)
    expectation_gradients_1 = expectation_gradients(group1, group2, grads_list , epsilons_access, eps_active, eps_nonactive , M_t)
    for key in grads_list[0]:
        total = 0
        for i,g in enumerate(grads_list) : 
            sum_j = 0 
            for j,g2 in enumerate(grads_list[:i]) :
                sum_j += g2[key] * monte_carlo_expectation_corr(j, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) #Here should be the variance of the stochastic gradient estimator instead of g[key]
            #print(f"length of dataset for client {i}: {client_lengths[i]}")
            total += (g[key])* (g[key] * monte_carlo_variance_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) + 2*
                                monte_carlo_expectation_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) * sum_j) #Here should be the variance of the stochastic gradient estimator instead of g[key]
        agg[key] = total
    expectation_gradients_1 = {k: v**2 for k, v in expectation_gradients_1.items()}
    diff_dict = {k: agg[k] - expectation_gradients_1[k] for k in agg if k in expectation_gradients_1}

    return diff_dict



def average_gradients_unbiaised(grads_list, epsilons , biases):
    agg = {}
    M = len(epsilons)
    for key in grads_list[0]:
        total = 0
        for i, g in enumerate(grads_list):

            #bias1 = real_expectation(i, grads_list, epsilons)
            bias = biases[i] #monte_carlo_expectation(i,epsilons,10000 )
            total += (g[key]/ (M * bias)) # *(client_lengths[i]/sum(client_lengths))  #(g[key]/ bias2)  #
        
        agg[key] = total #/ M_t
    return agg

def average_gradients_unbiaised_corr(grads_list,group1,group2, biases):#,client_lengths):
    agg = {}   
    M = len(group1) + len(group2) #+ len(group3)
    for key in grads_list[0]:
        total = 0
        for i, g in enumerate(grads_list):

            bias = biases[i] #monte_carlo_expectation_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) #Monte carlo estimation of the bias
            total += (g[key] / (M * bias))#* (client_lengths[i]/sum(client_lengths)) #
            
        agg[key] = total #/ M_t #/ len(grads_list)
    return agg#,variance_gradients(group1,group2,grads_list,epsilons_access,eps_active,eps_nonactive,M_t)


def average_gradients_unbiaised_3groups_corr(grads_list,group1,group2,group3, biases):#,client_lengths):
    agg = {}   
    M = len(group1) + len(group2) + len(group3)
    for key in grads_list[0]:
        total = 0
        for i, g in enumerate(grads_list):

            bias = biases[i] #monte_carlo_expectation_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) #Monte carlo estimation of the bias
            total += (g[key] / (M * bias))#* (client_lengths[i]/sum(client_lengths)) #
            
        agg[key] = total #/ M_t #/ len(grads_list)
    return agg

def average_gradients_unbiaised_4groups_corr(grads_list,group1,group2,group3,group4, biases):#,client_lengths):
    agg = {}   
    M = len(group1) + len(group2) + len(group3) + len(group4)
    for key in grads_list[0]:
        total = 0
        for i, g in enumerate(grads_list):

            bias = biases[i] #monte_carlo_expectation_corr(i, group1, group2, epsilons_access, eps_active, eps_nonactive, N=10000) #Monte carlo estimation of the bias
            total += (g[key] / (M * bias))#* (client_lengths[i]/sum(client_lengths)) #
            
        agg[key] = total #/ M_t #/ len(grads_list)
    return agg

def apply_gradient(model, avg_grad, lr):
    with torch.no_grad():
        for name, param in model.named_parameters():
            #print(f"lr : {lr} , name : {name} , avg_grad[name] : {avg_grad[name]}")
            param -= lr * avg_grad[name] 

def apply_gradient_normalized(model, avg_grad, lr):
    avg_grad_array = np.concatenate([v.ravel() for v in avg_grad.values()])
    norm = np.linalg.norm(avg_grad_array)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param -= lr * avg_grad[name] / norm

#Clipping
def apply_gradient_clipped(model, avg_grad, lr, max_norm=1.0):
    #Calculating the L2 norm of the total gradient
    total_norm = torch.sqrt(sum((g.norm() ** 2 for g in avg_grad.values())))
    clip_coef = min(1.0, max_norm / (total_norm + 1e-6))  

    with torch.no_grad():
        for name, param in model.named_parameters():
            param -= lr * (avg_grad[name] * clip_coef)
