import gurobipy
import numpy as np
import time

# MILP
# initialize graph and demand
node_num = 8
chunk_num = 8
chunk_num_list = np.ones(node_num)*chunk_num
# adjacency_matrix = np.zeros((node_num, node_num))
adjacency_matrix = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
                             [1, 0, 1, 1, 0, 1, 0, 0],
                             [1, 1, 0, 1, 0, 0, 1, 0],
                             [1, 1, 1, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 0, 1, 0, 1, 1],
                             [0, 0, 1, 0, 1, 1, 0, 1],
                             [0, 0, 0, 1, 1, 1, 1, 0]])

# capacity_matrix = np.zeros((node_num, node_num))  # chunk/s
capacity_matrix = adjacency_matrix*(0.25e7) # (0.5e7)  # (1e7)

# AlltoAll
demand_matrix = np.zeros((node_num, node_num, chunk_num))

for i in range(chunk_num):
    a = np.zeros((node_num))
    a[i] = 1
    demand_matrix[:,:,i] = np.tile(a, (node_num, 1))

# demand_matrix = np.expand_dims(demand_matrix, axis=2)
# demand_matrix = np.expand_dims(demand_matrix,2).repeat(chunk_num,axis=2)


tau = 0.6e-6 # 0.4e-6  # epoch duration
alpha_matrix = np.ones((node_num, node_num))*(0.6e-6)
delta_matrix = alpha_matrix/tau
delta_ceil_matrix = np.ceil(delta_matrix)

epoch_num = 10

actual_capacity_matrix = capacity_matrix * tau


# LP
model = gurobipy.Model('LP')
F = model.addVars(node_num, node_num, node_num, epoch_num, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='LinkChunk')
B = model.addVars(node_num, node_num, epoch_num, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='BufferChunk')
R = model.addVars(node_num, node_num, epoch_num, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='ReceiveChunk')  # gurobipy.GRB.BINARY
R_prime = model.addVars(node_num, node_num, epoch_num, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='ReceiveChunk')  # gurobipy.GRB.BINARY


# for i in range(node_num):
#     for j in range(node_num):
#         if j != i:
#             model.addConstr(B[i,j,0] == 0)

# for i in range(node_num):
#     for k in range(epoch_num):
#         model.addConstr(R[i,i,k] == 0)

for i in range(node_num):
    for j in range(node_num):
        for k in range(epoch_num):
            model.addConstr(gurobipy.quicksum(F[s, i, j, k] for s in range(node_num)) <= capacity_matrix[i,j] * tau, name='Capacity constraints')

for s in range(node_num):
    for n in range(node_num):
        neighbor_node_from_list = np.nonzero(adjacency_matrix[:, n])[0]
        neighbor_node_to_list = np.nonzero(adjacency_matrix[n,:])[0]
        for k in range(1, epoch_num-1):  # k?
            neighbor_from_list = []
            for i in neighbor_node_from_list:
                if k - delta_ceil_matrix[i,n] >= 0:
                    neighbor_from_list.append(i)  # i to n to j
            model.addConstr(B[s,n,k] + gurobipy.quicksum(F[s, i, n, int(k-delta_ceil_matrix[i, n])] for i in neighbor_from_list) == B[s,n,k+1] + R_prime[s,n,k] + gurobipy.quicksum(F[s, n, j, k+1] for j in neighbor_node_to_list), name='Flow conservation constraints')

for s in range(node_num):
    for n in range(node_num):
        neighbor_node_from_list = np.nonzero(adjacency_matrix[:, n])[0]
        for k in range(1, epoch_num):
            neighbor_from_list = []
            for i in neighbor_node_from_list:
                if k - delta_ceil_matrix[i, n]-1 >= 0:
                    neighbor_from_list.append(i)
            model.addConstr(B[s,n,k] == B[s,n,k-1] + gurobipy.quicksum(F[s,i,n,int(k-delta_ceil_matrix[i,n]-1)] for i in neighbor_from_list), name='Buffer constraints')

for s in range(node_num):
    for d in range(node_num):
        for k in range(epoch_num):
            model.addConstr(R[s,d,k] == gurobipy.quicksum(R_prime[s,d,r] for r in range(k)))
        model.addConstr(R[s,d,epoch_num-1] == gurobipy.quicksum(demand_matrix[s,d,c] for c in range(int(chunk_num_list[s]))), name='Demand constraints')

model.setObjective(gurobipy.quicksum(1/(k+1) * R[s,d,k] for k in range(epoch_num) for s in range(node_num) for d in range(node_num)), gurobipy.GRB.MAXIMIZE)


model.Params.MIPGap = 0.01
start_time = time.perf_counter()
model.optimize()
end_time = time.perf_counter()

execution_time = end_time - start_time

model.write('LP.lp')
