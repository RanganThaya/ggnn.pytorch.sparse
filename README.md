# PyTorch Implementation of GGNN for Sparse Graphs

Main Code Obtained from: https://github.com/JamesChuanggg/ggnn.pytorch

The original code is done for a dense graph. But converting the data to Adjacency Matrix. This was not memory efficient for large sparse graphs. This implementation addreses that problem.

This model accepts Adjacency List in the form of Edge_Num -> [src1, dest1], [src2,dest2] ...
