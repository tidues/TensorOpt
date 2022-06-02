# A Gurobi Wrapper that Supports Numpy Operations

Module `tensorgrb.py` fully supports constructing, combining, manipulating Gurobi variables and constraints using numpy arrays. This interface is convenient for setting up matrix-based formulations. However, for super-large-size models, the current version is slow. 


The other module `tensorgp.py` uses Gurobi's native matrix api. The construction time is much faster than `tensorgrb,py`, but the matrix operations are very limited.

## Module `tensorgrb.py` 
Suppose we want to construct the following formulation, where $\langle \cdot, \cdot \rangle$ is the Frobenius inner product, matrices $A, B, C, X, Y$ are of shapes $(m, n), (m, m), (m, n), (m, n), (m, m)$.
$$
\begin{align}
\min & \quad \langle C, X \rangle + \text{trace}(BY)\\
\text{s.t.} & \quad A X^\intercal \leq B^\intercal Y\\
            & \quad X, Y \geq 0.
\end{align}
$$
We can use the following code.
    from tensorgrb import Model
    import numpy as np
    
    # initalize input parameters
    m = 5
    n = 6
    A = np.random.random((m, n))
    B = np.random.random((m, m))
    C = np.random.random((m, n))

    # initialize model
    md = Model('test')

    # add variables
    X = md.var((m, n), lb=0, name='X')
    Y = md.var((m, m), lb=0, name='Y')

    # add constraints
    md.con(A @ X.T, '<=', B.T @ Y)

    # add objective
    md.obj((C * X).sum() + np.trace(B @ Y), minimize=True)

    # solve model
    print(md.solve())

Note that the variables $X$ and $Y$ are essentially numpy arrays, so most numpy functions can be directly applied to them.
