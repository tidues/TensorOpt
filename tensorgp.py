import numpy as np
import gurobipy as gp
import warnings
from itertools import product
from types import GeneratorType


#### generic model distributor
class Model:
    def __init__(self):
        pass

    def __new__(cls, solver='gurobi', **args):
        solvers = {
            'gurobi': GrbModel,
            'cplex': GrbModel
        }
        return solvers[solver](**args)


# main model
class GrbModel:
    def __init__(self, name="", grb_display=0, **args):
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', grb_display)
        self.env.start()
        self.name = name
        self.md = gp.Model(name, env=self.env)
        self.typemap = {
            "C": gp.GRB.CONTINUOUS, 
            "B": gp.GRB.BINARY,
            "I": gp.GRB.INTEGER
        }
        self.statusCode = {
            gp.GRB.OPTIMAL: 'OPTIMAL', 
            gp.GRB.INFEASIBLE: 'INFEASIBLE', 
            gp.GRB.INF_OR_UNBD: 'INF_OR_UNBD', 
            gp.GRB.UNBOUNDED: 'UNBOUNDED', 
        }
        self.varidx = 0
        self.vars = {}
        self.varsidx = {}
        self.conidx = 0
        self.cons = {}
        self.considx = {}

    def var_val(self, mvars):
        return mvars.getAttr('x')

    def obj_val(self):
        return self.md.objVal

    def var(self, shape, lb=-float('inf'), ub=float('inf'), vtype='C', name=""):
        # create the array of vars
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        elif type(shape) is not tuple:
            shape = tuple(shape)
        if name is None or name == "":
            params = {'lb':lb, 'ub':ub, 'vtype':vtype}
            res = self.md.addMVar(shape, **params)
        else:
            params = {'lb':lb, 'ub':ub, 'vtype':vtype, 'name':name}
            res = self.md.addMVar(shape, **params)
            # only save the named variables
            self.vars[name] = res
        return res

    def idxNameDict(self, items, itemType='var'):
        res = {}
        if isinstance(items, gp.tupledict):
            for key in items:
                if itemType == 'var':
                    res[key] = items[key].varName
                else:
                    if isinstance(items[key], gp.QConstr):
                        res[key] = items[key].QCName
                    else:
                        res[key] = items[key].constrName
        else:
            if itemType == 'var':
                items = items.varName
            else:
                items = items.constrName
            it = np.nditer(items, flags=['multi_index', 'refs_ok'])
            for x in it:
                key = it.multi_index
                res[it.multi_index] = str(x)
        return res

    # any array of expressions with compatible rhs array
    # sense "=", ">=", or "<="
    def con(self, exprs, sense, rhs, name=""):
        if sense == '=':
            exprs = exprs == rhs
        elif sense == '>=':
            exprs = exprs >= rhs
        elif sense == '<=':
            exprs = exprs <= rhs
        if name is None or name == "":
            params = {}
        else:
            params = {'name': name}

        # add constraints
        if isinstance(exprs, GeneratorType):
            res = self.md.addConstrs(exprs, **params)
        else:
            res = self.md.addConstr(exprs, **params)
        if name is not None and name != "":
            # only save the named constraints
            self.cons[name] = res
        return res

    # update model
    # vars/cons-idx_update: update name list ['key1', 'key2']
    # []: no update
    # 'all': update all
    def update(self, varsidx_update=[], considx_update=[]):
        self.md.update()
        # update required considx and varsidx
        if varsidx_update == 'all':
            varsidx_update = self.vars.keys()
        for name in varsidx_update:
            self.varsidx[name] = self.idxNameDict(self.vars[name], itemType='var')
        if considx_update == 'all':
            considx_update = self.cons.keys()
        for name in considx_update:
            self.considx[name] = self.idxNameDict(self.cons[name], itemType='con')

    # set gurobi parameters
    def setParams(self, params):
        for key in params:
            self.md.setParam(key, params[key])

    # objective function
    def obj(self, expr, sense='min'):
        if type(expr) is np.ndarray:
            expr = expr[0]
        if sense == 'min':
            sense = gp.GRB.MINIMIZE
        else:
            sense = gp.GRB.MAXIMIZE
        self.md.setObjective(expr, sense)

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
            if status in self.statusCode:
                return self.statusCode[status]
            else:
                return 'STATUS CODE: ' + str(status)
