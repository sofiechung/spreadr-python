# spreadr-python
This package is the Python version of Cynthia Siew's spreadr package, which is written in R. Documentation and source code can be found here: https://github.com/csqsiew/spreadr 

The notion of spreading activation is a prevalent metaphor in the cognitive sciences. This package provides the tools for cognitive scientists and psychologists to conduct computer simulations that implement spreading activation in a network representation. The algorithmic method implemented in spreadr subroutines follows the approach described in Vitevitch, Ercal, and Adagarla (2011, Frontiers), who viewed activation as a fixed cognitive resource that could spread among nodes that were connected to each other via edges or connections (i.e., a network). See Vitevitch, M. S., Ercal, G., & Adagarla, B. (2011).

# getting started
## prerequisites
There are also a few Python libraries that need to be installed, in order for the package to function: NumPy, Pandas, and NetworkX. They can all be installed with pip by executing the pip install command as shown below:

```
pip install numpy
```
```
pip install pandas
```
```
pip install networkx
```
If the above commands do not work, try replacing "pip" with "pip3". 
## installation
spreadr-python repo can be cloned:
```
git clone https://github.com/sofiechung/spreadr-python.git
```
Alternatively, source files of the most recent commit can just be directly downloaded.

# spreadr function
## arguments
**network**: A database which represents the nodes and edges of the network that spreading activation will be applied to. Can be a Pandas dataframe, NumPy ndarray, or NetworkX graph. For example, if one is using a NumPy ndarray as their network representation, the input could look like this:
```
example_network = np.array([[0,0,0,1]],
                            [1,0,0,1],
                            [0,0,0,1],
                            [0,1,0,0],
```
The above example is a 4x4 adjacency matrix which illustrates a connection between (node1,node4), (node1,node2), (node2,node4), and (node3,node4).

**start_run**: A Pandas dataframe which contains the activation values assigned to specific nodes at time t=0. Here is an example input:
```
init_act = {'node':[0],'activation':[20]}
example_start_run = pd.Dataframe(data = init_act)
```
The above example indicates that at time t=0, we want to apply activation value 20 to node 0 in our network. 

**retention**: The proportion of activation retained in the originator node (ranges from 0 to 1). Default is 0.5. 

**time**: The number of times to run the spreading activation process. Default is 10. 

**threshold_to_stop**: An integer indicating how many time steps should spreading activation run for. Default is None.

**decay**: The proportion of activation lost at each time step (ranges from 0 to 1). Default is 0.

**suppress**: Nodes with activation values lower than this value will have their activations forced to 0. Typically this will be a very small value (e.g., < .001). Default is 0.

**include_t0**: A boolean where TRUE indcates that the output will include time t=0, and FALSE indicates that the ouput will not include time t=0. Default is False.

**never_stop**: A boolean where TRUE indicates for the program to continue running, even if there is an infinite loop, and FALSE indicates for the program to terminate once >10000 iterations have passed. Default is False.

## output
The spreadr function will return a Pandas dataframe of the network which maps each node to their activation value at every timestep. Here is an example output:
```
  node  activation  time
0    a        10.0     0
1    b         0.0     0
2    c         0.0     0
3    a         0.0     1
4    b         1.0     1
5    c         9.0     1
```
### modified output
The spreadr-python package includes two additional functions which modifies the original output of the spreadr function. These are especially useful for extremely large networks:

**extract_specific**: This function will print a specified node's activation value at a specified time.

**extract_all**: This function will print either all the activation values for a specified node or all the activation values for a specified time, or both.

### output notes
As is, the spreadr function's output rounds all activation values up to 10 decimal points, as shown below:
```
  d = {'node': nodes, 'activation': [round(elem,10) for elem in activations], 'time': times}
  return_df = pd.DataFrame(data=d)
```
This can obviosuly be modified to however the user sees fit, but just note that the test file **spreadr_test.py** will only pass all test cases with the above format. 

# roadmap
- [ ] Implement Cython package into **create_mat_t.py** to decrease runtime for extremely large networks.
- [ ] Create a function which outputs a visual simulation of the spreading activation for a network.
# 
