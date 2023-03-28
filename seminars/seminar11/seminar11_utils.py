import numpy as np

class StatsManager:

    @staticmethod
    def traverse(o, tree_types=(list, tuple, np.ndarray)):
        if isinstance(o, tree_types):
            for value in o:
                for subvalue in StatsManager.traverse(value, tree_types):
                    yield subvalue
        else:
            yield o

    def __init__(self, *names):
        self.stats = {}
        self.names = list(names)
        for name in names:
            self.stats[name] = []

    def add_stat_names(self, *names):
        self.names = self.names + list(names)
        for name in names:
            if name in self.stats.keys():
                raise Exception(
                    "name '{}' has already been presented!".format(name))
            self.stats[name] = []

    def add_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                if vals[0] is not None:
                    self.add(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                if val is not None:
                    self.add(name, val)
            return
        raise Exception('stats update is ambiguous')

    def add(self, name, val):
        self.stats[name][-1] += val
    
    def upd_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                if vals[0] is not None:
                    self.upd(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                if val is not None:
                    self.upd(name, val)
            return 
        raise Exception('stats update is ambiguous')
    
    def upd(self, name, val):
        self.stats[name].append(val)
    
    def get(self, name):
        return self.stats[name]
    
    def draw(self, axs, names=None):
        axs_list = list(self.traverse(axs))
        if names is None:
            names = self.names
        for i, name in enumerate(names):
            axs_list[i].plot(self.get(name))
            axs_list[i].set_title(name)
    
    def reset(self):
        for name in self.stats.keys():
            self.stats[name] = []