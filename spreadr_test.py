import sys
import pytest
import spreadr
import pandas as pd
import numpy as np
import networkx as nx

def test_np_directed_edges():
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,0,0,1,0],
                        [1,0,0,1,0],
                        [0,0,0,1,0],
                        [0,1,0,0,0],
                        [1,1,0,0,0]])
    result = spreadr.spreadr(network, start_run, retention=0, time=5, suppress=0, decay=0, include_t0=True)
    expected = pd.DataFrame({'node':[0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],
                             'activation':[20.,0.,0.,0.,0.,0.,0.,0.,20.,0.,0.,20.,0.,0.,0.,10.,0.,0.,10.,0.,0.,10.,0.,10.,0.,5.,10.,0.,5.,0.],
                             'time':[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]})
    assert expected.equals(result)

def test_np_weighted_edges():
    d = {'node':[0], 'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,9],
                       [1,0,0],
                       [9,0,0]])
    result = spreadr.spreadr(network, start_run, retention=0, time=1, include_t0=True)
    expected = pd.DataFrame({'node':[0,1,2,0,1,2], 
                            'activation':[10.,0.,0.,0.,1.,9.],
                            'time':[0,0,0,1,1,1]})
    assert expected.equals(result)

def test_np_retention_1():
    d = {'node':[0], 'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,1],
                        [1,0,0],
                        [1,0,0]])
    result = spreadr.spreadr(network, start_run, retention=0.5,time=1, include_t0=True)
    expected = pd.DataFrame({'node':[0,1,2,0,1,2],
                             'activation':[10.,0.,0.,5.,2.5,2.5],
                             'time':[0,0,0,1,1,1]})
    assert expected.equals(result)

def test_np_retention_2():
    d = {'node':[0], 'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,1],
                        [1,0,0],
                        [1,0,0]])   
    result = spreadr.spreadr(network, start_run, retention=0.8, time=2, include_t0=True)
    expected = pd.DataFrame({'node':[0,1,2,0,1,2,0,1,2],
                             'activation':[10.,0.,0.,8.,1.0,1.0,6.8,1.6,1.6],
                             'time':[0,0,0,1,1,1,2,2,2]})
    for i,j in zip(list(result['activation']), list(expected['activation'])):
        if i != j:
            print(i,j)
    assert expected.equals(result)

def test_np_decay():
    d = {'node':[0], 'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,1],
                        [1,0,0],
                        [1,0,0]])   
    result = spreadr.spreadr(network, start_run, retention=0.8, time=2, decay=0.1, include_t0=True)
    result_sums = np.add.reduceat(list(result['activation']), np.arange(0, len((result['activation'])),3))
    expected_sums = [10., 9., 8.1]
    assert (expected_sums and result_sums).all()

def test_np_suppress():
    d = {'node':[0], 'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,1],
                        [1,0,0],
                        [1,0,0]])  
    not_suppress = spreadr.spreadr(network, start_run, retention=0.8, time=3, include_t0=True)
    suppress = spreadr.spreadr(network, start_run, retention=0.8, suppress=2, time=2, include_t0=True)
    for i,j in zip(not_suppress['activation'], suppress['activation']):
        if i < 2:
            assert j == 0
    
def test_np_leafnodes():
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1,0,0,1],
                        [0,0,0,1,1],
                        [0,0,0,0,0],
                        [1,1,1,0,0],
                        [0,0,0,0,0]])
    result = spreadr.spreadr(network, start_run, retention=0, time=3, suppress=0, decay=0, include_t0=True)
    expected = pd.DataFrame({'node':[0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4],
                             'activation':[20.,0.,0.,0.,0.,0.,10.,0.,0.,10.,0.,0.,0.,5.,15.,1.666667,1.666667,1.666667,0.,15.],
                             'time':[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]})
    
    assert expected.equals(result)

def test_pd_weighted():
    d = {'node':['a'],'activation':[10]}
    start_run = pd.DataFrame(data=d)
    network = pd.DataFrame([[0,1,9],[1,0,0],[9,0,0]], columns = ['a','b','c'], index = ['a','b','c'])
    result = spreadr.spreadr(network, start_run, retention=0, time=1, include_t0=True)
    expected = pd.DataFrame({'node':['a','b','c','a','b','c'],
                             'activation':[10.,0.,0.,0.,1.,9.],
                             'time':[0,0,0,1,1,1]})
    assert expected.equals(result)

def test_nx_weighted():
    d = {'node':['a'],'activation':[10]}
    start_run = pd.DataFrame(data=d)
    nodes = ['a','b','c']
    edges = [('a','b'),('a','c'),('b','a'),('c','a')]
    network = nx.Graph()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)
    nx.set_edge_attributes(network, {('a','b'):{'weight':1},('a','c'):{'weight':9},('b','a'):{'weight':1},('c','a'):{'weight':9}})
    result = spreadr.spreadr(network, start_run, retention=0, time=1, include_t0=True)
    expected = pd.DataFrame({'node':['a','b','c','a','b','c'],
                             'activation':[10.,0.,0.,0.,1.,9.],
                             'time':[0,0,0,1,1,1]})
    assert expected.equals(result)

def test_stops_if_infinite():
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    pytest.raises(RuntimeError, spreadr.spreadr, network, start_run, retention=0, time=None, threshold_to_stop=1)

def test_stop_cond1():
    """error thrown if both time, threshold_to_stop is None"""
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    pytest.raises(AssertionError, spreadr.spreadr, network, start_run, time=None)  

def test_stop_cond2():
    """terminates with time only"""
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    result = spreadr.spreadr(network, start_run, time=1)  
    expected = pd.DataFrame({'node':[0,1],
                             'activation':[10.,10.],
                             'time':[1,1]})
    assert expected.equals(result)

def test_stop_cond3():
    """terminates with threshold_to_stop only"""
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    result = spreadr.spreadr(network, start_run, time=None, threshold_to_stop=2.5, decay=0.5)
    expected = pd.DataFrame({'node':[0,1,0,1,0,1],
                             'activation':[20/4, 20/2/2, 20/4/4, (20/2/2 + 20/4/2)/2, 20/4/4/4, (((20/2/2 + 20/4/2)/2 + 20/4/4/2)/2)],
                             'time':[1,1,2,2,3,3]}) 
    assert expected.equals(result)

def test_stop_cond4():
    """terminates with both time and threshold_to_stop at threshold"""
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    result = spreadr.spreadr(network, start_run, time=6, threshold_to_stop=2.5, decay=0.5)
    expected = pd.DataFrame({'node':[0,1,0,1,0,1],
                             'activation':[20/4, 20/2/2, 20/4/4, (20/2/2 + 20/4/2)/2, 20/4/4/4, (((20/2/2 + 20/4/2)/2 + 20/4/4/2)/2)],
                             'time':[1,1,2,2,3,3]})     
    assert expected.equals(result)

def test_stop_cond5():
    """terminates with both time and threshold_to_stop at time"""
    d = {'node':[0], 'activation':[20]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    result = spreadr.spreadr(network, start_run, time=4, threshold_to_stop=1,)
    expected = pd.DataFrame({'node':[0,1,0,1,0,1,0,1],
                             'activation':[10.,10.,5.,15.,2.5,17.5,1.25,18.75],
                             'time':[1,1,2,2,3,3,4,4]})     
    assert expected.equals(result)    

def test_start_run1():
    """start_run with time column adds activation at the right times"""
    d = {'node':[0,0], 'activation':[10,10], 'time':[1,2]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    result = spreadr.spreadr(network, start_run, time=3)
    expected = pd.DataFrame({'node':[0,1,0,1,0,1], 
                             'activation':[10.,0.,15.,5.,7.5,12.5],
                             'time':[1,1,2,2,3,3]})
    assert expected.equals(result)

def test_start_run2():
    """error thrown if nodes specified in start_run do not exist in network"""
    d = {'node':[0,'dne'], 'activation':[10,10], 'time':[1,2]}
    start_run = pd.DataFrame(data=d)
    network = np.array([[0,1],
                        [0,0]])
    pytest.raises(KeyError, spreadr.spreadr, network, start_run, time=3)


if __name__ == "__main__":
    import sys
    res = pytest.main(["-k", " or ".join(sys.argv[1:]), "-v", __file__])