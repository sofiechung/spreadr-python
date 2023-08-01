import pandas as pd
import numpy as np
import networkx as nx
from create_mat_t import *
from decimal import *

def spreadr(network, start_run, retention=0.5, time=10, threshold_to_stop=None, decay=0, suppress=0, include_t0=False, never_stop=False):
    # is network a networkx graph? is network a numpy ndarray/pandas dataframe and if so, is it square-like?
    assert isinstance(network, np.ndarray) or isinstance(network, pd.DataFrame) or isinstance(network, nx.Graph)
    if isinstance(network, (np.ndarray, pd.DataFrame)):
        assert network.ndim ==2 and network.shape[0] == network.shape[1]
        # set n_nodes
        n_nodes = network.shape[0]
    else:
        # set n_nodes for networkx graph
        n_nodes = network.number_of_nodes()
    # is start_run in the correct format?
    assert isinstance(start_run, pd.DataFrame) and 'node' in start_run.columns and 'activation' in start_run.columns
    # time is a non-negative integer, if it is provided
    if 'time' in start_run.columns:
        assert all(x >= 0 for x in start_run['time']) and all(x % 1 == 0 for x in start_run['time'])
    # is decay a number between 0 and 1 inclusive?
    assert 0 <= decay <= 1
    # is retention an appropiate number or numeric vector
    if not isinstance(retention, list):
        assert isinstance(retention, (float,int)) and 0 <= retention <= 1
        # it is easier if we assume retention is always a vector
        retention = np.repeat(retention, n_nodes)
    else:
        assert isinstance(retention, list)
        assert len(retention) == n_nodes
        # assert np.all(0 <= retention[0] <= 1)
        assert all(0 <= x <= 1 for x in retention)
    # is return_type ok?
    assert isinstance(include_t0, bool)

    # are terminating conditions ok? (time and threshold_to_stop)
    assert not(time == None and threshold_to_stop == None), 'time and threshold_to_stop cannot both be None' #check truth conditions
    assert time == None or time % 1 == 0
    assert threshold_to_stop == None or threshold_to_stop > 0

    # if not already, convert network to pandas dataframe and add node names
    if isinstance(network, np.ndarray):
        network = pd.DataFrame(network, columns=[x for x in range(0, network.shape[1])], index=[x for x in range(0, network.shape[0])])
    elif isinstance(network, nx.Graph):
        network = nx.to_pandas_adjacency(network, weight='weight')
        assert isinstance(network, pd.DataFrame)

    # is there any node in start_run which does not exist in network?
    not_exist_nodes = list(set(list(start_run['node'])).difference(set(list(network.columns))))
    if len(not_exist_nodes) != 0:
        raise KeyError(f"These nodes specified in start_run don't exist in network: {not_exist_nodes} ")
    
    # variables in the loop:
    #    d: numeric integer vector, the degree of each node
    #  a_t: numeric vector, the activation at time t
    #  mat: base matrix version of network
    degree = np.array(network.sum(axis=1)) # row sums
    n_nodes = len(degree)
    a_t = np.repeat([0], n_nodes) # empty vector
    mat = network.to_numpy()

    # pre-loop set-up: if 'time' not in start_run.columns, start_run specifies the activation value
    # of certain nodes at t=0. Otherwise, start_run specifies when to add how much 
    # activation to certain nodes, so check for activations to be added at t=0.
    if not('time' in start_run.columns):
        sr_time = [0 for x in range(0,len(start_run.index))] 
        start_run['time'] = sr_time
   
    
    current_time, activations = 0, None
    while True:
        # spreading activation
        a_tm1 = a_t
        mat_t = create_mat_t(mat, a_tm1, degree, retention)
        # create_mat_t creates an "activation matrix", where each (i, j) entry is 
        # the activation at time t, at node j, due to node i
        a_t = mat_t.sum(axis=0)
        
        # decay and suppress
        a_t = a_t * (1-decay) 
        a_t[a_t < suppress] = 0
       
        # apply start_run instructions
        start_run_t = start_run[start_run['time'] == current_time]

        for i in range(0, len(start_run_t.index)+1):
            try:
                j = network.columns.get_loc(start_run_t['node'][i]) 
            except KeyError:
                continue
            a_t[j] = a_t[j] + start_run_t['activation'][i]
        
        # record
        if activations == None:
            activations = []
            activations.extend(a_t)
        else:
            activations.extend(a_t)
       
        # check termination
        if not(time == None) and current_time >= time: 
            break
        if not(threshold_to_stop==None) and all(x < threshold_to_stop for x in a_t):
            break
        if time == None and decay == 0 and not(never_stop) and current_time > 10000:
            raise RuntimeError('Stopping because there might potentially be an infinite loop. Set never_stop=True to override.')
        current_time += 1
    
    nodes = list(network.columns) * (current_time+1)
    times = np.repeat([x for x in range(0,current_time+1)], n_nodes)
    d = {'node': nodes, 'activation': [f'{elem:.18f}' for elem in activations], 'time': times}
    return_df = pd.DataFrame(data=d)

    if not(include_t0):
        return_df = return_df[return_df['time'] != 0]
        if len(return_df.index) != 0:
            return_df.rename(index = lambda x: x-n_nodes, inplace=True) #rename indices so that they start at 0 

    return return_df


if __name__ == '__main__':
    pass
    # d = {'node': [1,4,7],'activation':[100,100,100],'time':[0,0,0]}
    # start_run = pd.DataFrame(data=d)
    # network = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 0 ,0],
    #            [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #            [0, 0, 0, 0, 0, 0, 0, 0, 0]]) 
    # retention = [0,0,0,0,0.5,0,0,1,0]
    # result = spreadr(network, start_run, time=5, retention=retention, include_t0=True)
    # print(result.head)
    # d = {'node':['a','c'],'activation':[10,5],'time':[3,2]}
    # start_run = pd.DataFrame(data=d)
    # nodes = ['a','b','c']
    # edges = [('a','b'),('a','c'),('b','a'),('c','a')]
    # network = nx.Graph()
    # network.add_nodes_from(nodes)
    # network.add_edges_from(edges)
    # nx.set_edge_attributes(network, {('a','b'):{'weight':1},('a','c'):{'weight':9},('b','a'):{'weight':1},('c','a'):{'weight':9}})
    # result = spreadr(network, start_run, retention=0, time=5, include_t0=True)
    # print(result)
    # case 1: directed edges
    d = {'node':[0],'activation':[20]}
    df = pd.DataFrame(data=d)
    mat = np.array([[0,0,0,1,0],
                    [1,0,0,1,0],
                    [0,0,0,1,0],
                    [0,1,0,0,0],
                    [1,1,0,0,0]])
    results = spreadr(mat, df, retention=0, time=5, suppress=0, decay=0, include_t0=True)
    print(f"case 1: directed edges \n{results}")
# case 2: weighted edges
    d = {'node':[0],'activation':[10]}
    df = pd.DataFrame(data=d)
    mat = np.array([[0,1,9],[1,0,0],[9,0,0]])
    results = spreadr(mat, df, retention=0, time=1, include_t0=True)
    print(f"case 2: weighted edges \n{results}")
# case 3: retention tests ----
    d = {'node':[0],'activation':[10]}
    df = pd.DataFrame(data=d)
    mat = np.array([[0,1,1],[1,0,0],[1,0,0]])
    results_1 = spreadr(mat, df, retention=0.5, time=1, include_t0=True)
    print(f"case 3.1: retention \n{results_1}") # expected result: node 1 should retain 50% of original activation
    results_2 = spreadr(mat, df, retention=0.8, time=2, include_t0=True)
    print(f"case 3.2: retention \n{results_2}") # expected result: node 1 should retain 80% of original activation
# case 4: decay test ----
    d = {'node':[0],'activation':[10]}
    df = pd.DataFrame(data=d)
    mat = np.array([[0,1,1],[1,0,0],[1,0,0]])
    results = spreadr(mat, df, retention=0.8, time=2, decay=0.1, include_t0=True)
    print(f"case 4: decay \n{results}") # expected result: total activation should decrease by a factor of 0.1 at each time point
# case 5: suppress test ----
    d = {'node':[0],'activation':[10]}
    df = pd.DataFrame(data=d)
    mat = np.array([[0,1,1],[1,0,0],[1,0,0]])
    results = spreadr(mat, df, retention=0.8, time=2, suppress=2, include_t0=True)
    print(f"case 5: suppress \n{results}")
# case 6: leaf nodes ----
    d = {'node':[0],'activation':[20]}
    df = pd.DataFrame(data=d)    
    mat = np.array([[0,1,0,0,1],[0,0,0,1,1],[0,0,0,0,0],[1,1,1,0,0],[0,0,0,0,0]])
    results = spreadr(mat, df, retention=0, time=3, suppress=0, decay=0, include_t0=True)
    print(f"case 6: leaf \n{results}")



