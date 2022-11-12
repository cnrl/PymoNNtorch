from pymonntorch.NetworkCore.Behavior import Behavior
import copy
import torch


def get_Recorder(variable):
    return Recorder(variables=[variable])


class Recorder(Behavior):
    visualization_module_outputs = []

    def __init__(self, variables, gap_width=0, tag=None, max_length=None, device='cpu'):
        super().__init__(tag=tag, variables=variables, gap_width=gap_width, max_length=max_length, device=device)
        
        self.add_tag('recorder')

        self.gap_width = self.get_init_attr('gap_width', 0)
        self.counter = 0
        self.new_data_available = False

        if type(variables) is str:
            variables = list(map(str.strip, variables.split(',')))

        self.variables = {}
        self.compiled = {}

        self.add_variables(self.get_init_attr('variables', []))
        self.reset()
        self.max_length = self.get_init_attr('max_length', None)

    def set_variables(self, object):
        assert self.device == object.device, "Recorder and object must be on the same device"
        self.reset()

    def add_variable(self, v):
        self.variables[v] = torch.tensor([], dtype=torch.float32, device=self.device)
        self.compiled[v] = None

    def add_variables(self, vars):
        for v in vars:
            self.add_variable(v)

    def find_objects(self, key):
        result = []
        if key in self.variables:
            result.append(self.variables[key])
        return torch.tensor(result, device=self.device)

    def reset(self):
        for v in self.variables:
            self.add_variable(v)

    def is_new_data_available(self):
        if self.new_data_available:
            self.new_data_available = False
            return True
        else:
            return False

    def get_data_v(self, variable, parent_obj):
        n = parent_obj  # used for eval string "n.x"
        s = parent_obj
        neurons = parent_obj
        synapse = parent_obj

        return copy.copy(eval(self.compiled[variable]))

    def save_data_v(self, data, variable):
        device = self.variables[variable].device
        self.variables[variable] = torch.concat([self.variables[variable], torch.tensor(data, device=device)])

    def new_iteration(self, parent_obj):
        if parent_obj.recording:
            self.counter += 1

            if self.counter >= self.gap_width:
                self.new_data_available = True
                self.counter = 0

                for v in self.variables:
                    if self.compiled[v] is None:
                        self.compiled[v] = compile(v, '<string>', 'eval')

                    data = self.get_data_v(v, parent_obj)
                    if data is not None:
                        self.save_data_v(data, v)

        if self.max_length is not None:
            self.cut_length(self.max_length)

    def cut_length(self, max_length):
        if max_length is not None:
            for v in self.variables:
                while len(self.variables[v]) > max_length:
                    self.variables[v] = self.variables[v][1:]

    def swapped(self, name):
        return self.swap(self.variables[name])

    def swap(self, x):
        return torch.swapaxes(x, 1, 0)

    def clear_recorder(self):
        print('clear')

        for v in self.variables:
            device = self.variables[v].device
            del self.variables[v]
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            self.variables[v] = torch.tensor([], dtype=torch.float32, device=device)

class EventRecorder(Recorder):

    def __init__(self, variables, tag=None, device='cpu'):
        super().__init__(variables, gap_width=0, tag=tag, max_length=None, device=device)

    def find_objects(self, key):

        result = []
        if key in self.variables:
            result.append(self.variables[key])

        if type(key) is str and key[-2:] == '.t' and key[:-2] in self.variables:
            result.append(self.variables[key[:-2]][:, 0])

        if type(key) is str and key[-2:] == '.i' and key[:-2] in self.variables:
            result.append(self.variables[key[:-2]][:, 1])

        return torch.tensor(result, device=self.device)

    def get_data_v(self, variable, parent_obj):
        n = parent_obj  # used for eval string "n.x"
        s = parent_obj
        neurons = parent_obj
        synapse = parent_obj

        data = eval(self.compiled[variable])
        indices = torch.where(data != 0)[0]

        if len(indices) > 0:
            result = []
            for i in indices:
                result.append([parent_obj.iteration, i])
            return torch.tensor(result, device=self.device)
        else:
            return None

    def save_data_v(self, data, variable):
        self.variables[variable] = torch.concat([self.variables[variable], data])
