"""MILP"""
import gurobipy
import numpy as np
import time


# ------------------------------initialize graph and demand-------------------------- #
node_num = 8
chunk_num = 8
chunk_num_list = np.ones(node_num)*chunk_num
# adjacency_matrix = np.zeros((node_num, node_num))
## NDv2
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

# -------------- collective mode
# AllReduce & AllGather
# demand_matrix = np.array([[0, 1, 1, 1, 1, 1, 1, 1],
#                           [1, 0, 1, 1, 1, 1, 1, 1],
#                           [1, 1, 0, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 0, 1, 1, 1, 1],
#                           [1, 1, 1, 1, 0, 1, 1, 1],
#                           [1, 1, 1, 1, 1, 0, 1, 1],
#                           [1, 1, 1, 1, 1, 1, 0, 1],
#                           [1, 1, 1, 1, 1, 1, 1, 0]])
# demand_matrix = np.expand_dims(demand_matrix,2).repeat(chunk_num,axis=2)

# AlltoAll & ReduceScatter
demand_matrix = np.zeros((node_num, node_num, chunk_num))
for i in range(chunk_num):
    a = np.zeros((node_num))
    a[i] = 1
    demand_matrix[:,:,i] = np.tile(a, (node_num, 1))

tau = 0.6e-6 # 0.4e-6  # epoch duration
alpha_matrix = np.ones((node_num, node_num))*(0.6e-6)
delta_matrix = alpha_matrix/tau
delta_ceil_matrix = np.ceil(delta_matrix)

epoch_num = 10

actual_capacity_matrix = capacity_matrix * tau

# ------------------------------initialize gurobi model------------------------------ #
model = gurobipy.Model('MILP')
F = model.addVars(node_num, node_num, node_num, epoch_num, chunk_num, lb=0, vtype=gurobipy.GRB.INTEGER, name='LinkChunk')
B = model.addVars(node_num, node_num, epoch_num, chunk_num, lb=0, vtype=gurobipy.GRB.INTEGER, name='BufferChunk')
R = model.addVars(node_num, node_num, epoch_num, chunk_num, lb=0, vtype=gurobipy.GRB.INTEGER, name='ReceiveChunk')  # gurobipy.GRB.BINARY
D = model.addVars(node_num, node_num, chunk_num, lb=0, vtype=gurobipy.GRB.INTEGER, name='DemandChunk')  # gurobipy.GRB.BINARY

# ------------------------------constraints------------------------------ #
for i in range(node_num):
    for j in range(node_num):
        if j != i:
            for c in range(int(chunk_num_list[i])):
                model.addConstr(B[i,j,0,c] == 0)

for i in range(node_num):
    for j in range(node_num):
        for k in range(epoch_num):
            model.addConstr(gurobipy.quicksum(F[s, i, j, k, c] for s in range(node_num) for c in range(int(chunk_num_list[s]))) <= capacity_matrix[i,j] * tau, name='Capacity constraints')

for s in range(node_num):
    for n in range(node_num):
        neighbor_node_from_list = np.nonzero(adjacency_matrix[:, n])[0]
        neighbor_node_to_list = np.nonzero(adjacency_matrix[n,:])[0]
        for k in range(1, epoch_num-1):
            neighbor_from_list = []
            for i in neighbor_node_from_list:
                if k - delta_ceil_matrix[i,n] >= 0:
                    neighbor_from_list.append(i)
            for c in range(int(chunk_num_list[s])):
                for j in neighbor_node_to_list: # i to n to j
                    model.addConstr(B[s,n,k,c] + gurobipy.quicksum(F[s, i, n, int(k-delta_ceil_matrix[i, n]), c] for i in neighbor_from_list) >= F[s, n, j, k+1, c], name='Flow conservation constraints')

for s in range(node_num):
    for n in range(node_num):
        neighbor_node_from_list = np.nonzero(adjacency_matrix[:, n])[0]
        for k in range(1, epoch_num):
            neighbor_from_list = []
            for i in neighbor_node_from_list:
                if k - delta_ceil_matrix[i, n] - 1 >= 0:
                    neighbor_from_list.append(i)
            for c in range(int(chunk_num_list[s])):
                model.addConstr(B[s,n,k,c] == B[s,n,k-1,c] + gurobipy.quicksum(F[s,i,n,int(k-delta_ceil_matrix[i,n]-1),c] for i in neighbor_from_list), name='Buffer constraints')


for s in range(node_num):
    for d in range(node_num):
        for c in range(int(chunk_num_list[s])):
            model.addConstr(D[s,d,c] == demand_matrix[s,d,c])

for s in range(node_num):
    for d in range(node_num):
        for c in range(int(chunk_num_list[s])):
            for k in range(epoch_num-1):
                model.addConstr(R[s,d,k,c] == gurobipy.min_(D[s,d,c], B[s,d,k+1,c]))
            model.addConstr(R[s,d,epoch_num-1,c] == demand_matrix[s,d,c], name='Demand constraints')

# ------------------------------objective------------------------------ #
model.setObjective(gurobipy.quicksum(1/(k+1) * R[s,d,k,c] for k in range(epoch_num-1) for s in range(node_num) for d in range(node_num) for c in range(int(chunk_num_list[s]))), gurobipy.GRB.MAXIMIZE)

model.Params.MIPGap = 0.01

start_time = time.perf_counter()
model.optimize()
end_time = time.perf_counter()

execution_time = end_time - start_time

transfer_time = tau * epoch_num

model.write('MILP.lp')
