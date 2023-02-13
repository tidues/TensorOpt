import numpy as np
import gurobipy as gp
import warnings
from stopwatch.stopwatch import Stopwatch
import sys
import docplex.mp.model as cp
from docplex.util.status import JobSolveStatus as jst

#### Core function to make np arrays of vars and constraints
# create array using provided function and parameters
def mkarr_h(size, func, params={}, idx=[]):
    if len(size) > 0:
        dim = size[0]
        res = []
        for tidx in range(dim):
            res.append(mkarr_h(size[1:], func, params=params, idx=idx + [tidx]))
        return res
    else:
        return func(idx, **params)

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


#### generic model distributor
class Model:
    def __init__(self):
        pass

    def __new__(cls, solver='gurobi', **args):
        solvers = {
            'gurobi': GrbModel,
            'cplex': CpxModel
        }
        return solvers[solver](**args)


#### the base model as interface and common functionalities
class BaseModel:
    def __init__(self, name=""):
        self.name = name
        self.md = self._gen_model()
        self.typemap, self.sensemap, self.statusmap, self.paramsmap = self._gen_maps()
        self.varidx = 0
        self.vars = {}
        self.conidx = 0
        self.cons = {}

    def varnum(self):
        res = 0
        for key in self.vars:
            res += self.vars[key].size
        return res

    def connum(self):
        res = 0
        for key in self.cons:
            res += self.cons[key].size
        return res

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

    # objective function
    def obj(self, expr, sense='min'):
        # if minimize:
        #     sense = gp.GRB.MINIMIZE
        # else:
        #     sense = gp.GRB.MAXIMIZE
        if type(expr) is np.ndarray:
            expr = expr[0]
        self._set_obj(self.sensemap[sense], expr)

    # solve
    def solve(self, params={}, timing=False, tname='time', withKey=True):
        try:
            for param, value in params.items():
                self.md.setParam(param, value)
        except (TypeError, ValueError):
            raise ValueError('Incorrect parameters or values.')  
        if timing:
            sw = Stopwatch()
            sw.init(start=True, name=tname)
            self._md_solve()
            sw.lap()
            res = self.obj_val(), sw.info(2, withKey=withKey)
        else:
            self._md_solve()
            if self.status() in self.statusmap['optimal'] or self.status() in self.statusmap['feasible']:
                res = self.obj_val()
            else:
                print("Optimization was stopped with status %d" % self.status())
                sys.exit(0)
        return res

    # get variable values
    def var_val(self, var):
        func_vec = np.vectorize(self._var_val)
        if type(var) is str:
            var = self.vars[var]
        return func_vec(var)

    # create single variable from parameters
    def _var_func(self, idx, lb, ub, vtype, name):
        if len(idx) == 0:
            name = name
        else:
            name = name+str(idx)
        if hasattr(lb, '__iter__'):
            lb = lb[tuple(idx)]
        if hasattr(ub, '__iter__'):
            ub = ub[tuple(idx)]
        return self._var_init(lb, ub, vtype, name)

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
        return self._con_init(con, name)

    #### public functions to be implemented
    # set gurobi parameters
    def setParams(self, params):
        for key in params:
            pkey, pth = self._gen_pkey(key)
            if pkey is None:
                return None
            elif len(self.paramsmap[key]) > 1:
                val = self.paramsmap[key][1][params[key]]
            else:
                val = params[key]
            self._set_params(pkey, val, pth)

    # get gurobi parameters
    def getParams(self, params):
        res = {}
        for key in params:
            pkey, pth = self._gen_pkey(key)
            if pkey is None:
                res[key] = None
            else:
                self._get_params(pkey, pth)
        return res

    def _gen_pkey(self, key):
        if isinstance(self.paramsmap[key][0], list):
            pth = self.paramsmap[key][0][:-1]
            pkey = self.paramsmap[key][0][-1]
        else:
            pkey = self.paramsmap[key][0]
            pth = None
        return pkey, pth

    def reset(self):
        pass

    def obj_val(self):
        pass

    def status(self):
        pass

    #### private functions to be implemented
    def _gen_model(self):
        pass

    # generate all common maps
    # typemap, sensemap, statusmap, paramsmap
    def _gen_maps(self):
        pass

    def _set_obj(self, sense, expr):
        pass

    def _md_solve(self):
        pass

    # get variable values
    def _var_val(self, var):
        return var.x

    def _var_init(self, lb, ub, vtype, name):
        pass

    def _con_init(self, con, name):
        pass


#### Gurobi Wrapper
# main model
class GrbModel(BaseModel):
    def __init__(self, name="", grb_display=0, **args):
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', grb_display)
        self.env.start()
        super().__init__(name=name) 

    def _set_params(self, pkey, val, *args):
        self.md.setParam(pkey, val)

    def _get_params(self, pkey, *args):
        return self.md.getParamInfo(pkey)

    def reset(self):
        self.md.reset()

    def obj_val(self):
        return self.md.objVal

    def status(self):
        return self.md.Status

    def _gen_model(self):
        return gp.Model(self.name, env=self.env)
    
    def _gen_maps(self):
        typemap = {
            "C": gp.GRB.CONTINUOUS, 
            "B": gp.GRB.BINARY,
            "I": gp.GRB.INTEGER
        }
        sensemap = {
            'min': gp.GRB.MINIMIZE,
            'max': gp.GRB.MAXIMIZE
        }
        statusmap = {
            'feasible': [7,8,9,10,13,15],
            'optimal': [2],
            'infeasible': [3],
            'unbounded': [5],
            'inf_or_unbd': [4]
        }
        paramsmap = {
            'presolve': ('Presolve',),
            'lp_reduce': ('DualReductions', {'primal': 0, 'dual': 1, 'both': 1, 'none': 0}),
            'lp_method': ('Method', {'auto': -1, 'primal': 0, 'dual': 1, 'network': -1, 'barrier': 2, 'sifting': -1, 'concurrent': 3, 'det_con': 4, 'det_con_slx': 5}),
            'slx_opt_tol': ('OptimalityTol',),
            'slx_fea_tol': ('FeasibilityTol',),
            'slx_mkz_tol': (None,),
            'slx_iter_lmt': ('IterationLimit',),
            'crossover': ('Crossover',),
            'bar_iter_lmt': ('BarIterLimit',),
            'bar_conv_tol': ('BarConvTol',),
            'time_lmt': ('TimeLimit',)
        }
        return typemap, sensemap, statusmap, paramsmap

    def _set_obj(self, sense, expr):
        self.md.setObjective(1 * expr, sense)

    def _md_solve(self):
        return self.md.optimize()

    # get variable values
    def _var_val(self, var):
        return var.x

    def _var_init(self, lb, ub, vtype, name):
        return self.md.addVar(lb=lb, ub=ub, vtype=self.typemap[vtype], name=name)

    def _con_init(self, con, name):
        return self.md.addConstr(con, name=name)


#### Cplex Wrapper
# main model
class CpxModel(BaseModel):
    def __init__(self, name="", **args):
        super().__init__(name=name) 

    def _set_params(self, pkey, val, pth):
        param_obj = self._get_param_obj(pth)
        setattr(param_obj, pkey, val)

    def _get_params(self, pkey, pth):
        param_obj = self._get_param_obj(pth)
        return getattr(param_obj, pkey)

    def _get_param_obj(self, pth):
        res = self.md.parameters
        for key in pth:
            res = getattr(res, key)
        return res

    def reset(self):
        pass

    def obj_val(self):
        return self.md.objective_value

    def status(self):
        return self.md.solve_status.value

    def _gen_model(self):
        return cp.Model(name=self.name)
    
    def _gen_maps(self):
        typemap = {
            "C": self.md.continuous_var,
            "B": self.md.binary_var,
            "I": self.md.integer_var
        }
        sensemap = {
            'min': 'min',
            'max': 'max'
        }
        statusmap = {
            'feasible': [1],
            'optimal': [2],
            'infeasible': [3],
            'unbounded': [4],
            'inf_or_unbd': [5]
        }
        paramsmap = {
            'presolve': (['preprocessing', 'presolve'],),
            'lp_reduce': (['preprocessing', 'reduce'], {'primal': 1, 'dual': 2, 'both': 3, 'none': 0}),
            'lp_method': (['lpmethod'], {'auto': 0, 'primal': 1, 'dual': 2, 'network': 3, 'barrier': 4, 'sifting': 5, 'concurrent': 6, 'det_con': 6, 'det_con_slx': 6}),
            'slx_opt_tol': (['simplex', 'tolerances', 'optimality'],),
            'slx_fea_tol': (['simplex', 'tolerances', 'feasibility'],),
            'slx_mkz_tol': (['simplex', 'tolerances', 'markowitz'],),
            'slx_iter_lmt': (['simplex', 'limits', 'iterations'],),
            'crossover': (['barrier', 'crossover'],),
            'bar_iter_lmt': (['barrier', 'limits', 'iteration'],),
            'bar_conv_tol': (['barrier', 'convergetol'],),
            'time_lmt': (['timelimit'],)
        }
        return typemap, sensemap, statusmap, paramsmap

    def _set_obj(self, sense, expr):
        self.md.set_objective(sense, 1 * expr)

    def _md_solve(self):
        return self.md.solve()

    # get variable values
    def _var_val(self, var):
        return self.md.solution.get_value(var)

    def _var_init(self, lb, ub, vtype, name):
        return self.typemap[vtype](name=name, lb=lb, ub=ub)

    def _con_init(self, con, name):
        return self.md.add_constraint(con, ctname=name)
