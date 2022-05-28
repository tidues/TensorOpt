import numpy as np
import gurobipy as gp
import warnings

# recursive: create array using provided function and parameters
# def mkarr_h(size, func, params={}, idx=[]):
#     if len(size) > 0:
#         dim = size[0]
#         res = []
#         for tidx in range(dim):
#             res.append(mkarr_h(size[1:], func, params=params, idx=idx + [tidx]))
#         return res
#     else:
#         return func(idx, **params)


# non_recursive: create array using provided function and parameters
def mkarr_h(size, func, params={}, idx=[]):
    if len(size) == 0:
        return func(idx, **params)
    arr = np.empty(tuple(size), dtype=object)
    with np.nditer(arr, flags=['multi_index', 'refs_ok'], op_flags=['writeonly']) as it:
        for x in it:
            x[...] = func(list(it.multi_index), **params)
    return arr


def mkarr(size, func, params={}):
    if type(size) is int:
        size = [size]
    elif type(size) is tuple:
        size = list(size)
    elif type(size) is list:
        size = size
    else:
        warnings.warn('Input "size" should be an integer or a tuple.')
    return np.array(mkarr_h(size, func, params=params))


# main model
class Model:
    def __init__(self, name=""):
        self.name = name
        self.md = gp.Model(name)
        self.typemap = {
            "C": gp.GRB.CONTINUOUS, 
            "B": gp.GRB.BINARY,
            "I": gp.GRB.INTEGER
        }
        self.status = {
            gp.GRB.OPTIMAL: 'OPTIMAL', 
            gp.GRB.INFEASIBLE: 'INFEASIBLE', 
            gp.GRB.INF_OR_UNBD: 'INF_OR_UNBD', 
            gp.GRB.UNBOUNDED: 'UNBOUNDED', 
        }
        self.varidx = 0
        self.vars = {}
        self.conidx = 0
        self.cons = {}

    def var(self, size=[], lb=-float('inf'), ub=float('inf'), vtype='C', name=""):
        if name is None or name == "":
            name = "var" + str(self.varidx)
            self.varidx += 1
        # create the array of vars
        params = {'lb':lb, 'ub':ub, 'vtype':vtype, 'name':name}
        self.vars[name] = mkarr(size, self._var_func, params=params)
        return self.vars[name]

    # any array of expressions with compatible rhs array
    # sense "=", ">=", or "<="
    def con(self, exprs, sense, rhs, name=""):
        if name is None or name == "":
            name = "con" + str(self.conidx)
            self.conidx += 1
        exprs = exprs - rhs
        params = {'exprs':exprs, 'sense':sense, 'name':name}
        if type(exprs) is np.ndarray:
            size = exprs.shape
        else:
            size = []
        self.cons[name] = mkarr(size, self._con_func, params=params)
        return self.cons[name]

    # add a set of constraints
    def conSet(self, cons):
        res = []
        for con in cons:
            res.append(self.con(*con))
        return res

    # set gurobi parameters
    def setParams(self, params):
        for key in params:
            self.md.setParam(key, params[key])

    # objective function
    def obj(self, expr, minimize=True):
        if minimize:
            sense = gp.GRB.MINIMIZE
        else:
            sense = gp.GRB.MAXIMIZE
        if type(expr) is np.ndarray:
            expr = expr[0]
        self.md.setObjective(1 * expr, sense)

    # solve
    def solve(self, display=0, params={}):
        self.md.setParam('LogToConsole', display)
        try:
            for param, value in params.items():
                self.md.setParam(param, value)
        except (TypeError, ValueError):
            raise ValueError('Incorrect parameters or values.')  
        self.md.optimize()
        status = self.md.status
        if status == 2:
            return self.md.objVal
        else:
            if status in self.status:
                return self.status[status]
            else:
                return 'STATUS CODE: ' + str(status)

    # get variable values
    def var_val(self, var):
        func = lambda var: var.x
        func_vec = np.vectorize(func)
        if type(var) is str:
            var = self.vars[var]
        return func_vec(var)

    # create single variable from parameters
    def _var_func(self, idx, lb, ub, vtype, name):
        if len(idx) == 0:
            name = name
        else:
            name = name+str(idx)
        return self.md.addVar(lb=lb, ub=ub, vtype=self.typemap[vtype], name=name)

    # create single constraint from parameters
    def _con_func(self, idx, exprs, sense, name):
        if len(idx) == 0:
            expr = exprs
            name = name
        else:
            expr = exprs[tuple(idx)]
            name = name+str(idx)
        if sense == '=' or sense == '==':
            con = expr == 0
        elif sense == '<=':
            con = expr <= 0
        elif sense == '>=':
            con = expr >= 0
        else:
            warnings.warn('Input "sense" should be =, >=, or <=.')
        return self.md.addConstr(con, name=name)
