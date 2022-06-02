# A Gurobi Wrapper that Supports Numpy Operations

Module `tensorgrb.py` fully supports constructing, combining, manipulating Gurobi variables and constraints using numpy arrays. This interface is convenient for setting up matrix-based formulations. However, for super-large-size models, the current version is slow. 


The other module `tensorgp.py` uses Gurobi's native matrix api. The construction time is much faster than `tensorgrb,py`, but the matrix operations are very limited.

## Module `tensorgrb.py` 
The follow is the formulation to be constructed.

$$
\begin{align}
\min & \langle C, X \rangle + \text{diag}(B, Y)\\
\test{s.t.} & A X^\intercal \leq B^\intercal Y\\
            &  X, Y \geq 0.
\end{align}
$$
