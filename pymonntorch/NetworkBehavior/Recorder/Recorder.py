"""
This module includes functions and classes to facilitate recording of network object \
variables through out the simulation time.
"""
import copy
import torch
from pymonntorch.NetworkCore.Behavior import Behavior


def get_Recorder(variable):
    """Returns a `Recorder` instance for the given variable.

    Args:
        variable (str): Name of the variable to record.

    Returns:
        Recorder: The `Recorder` object.
    """
    return Recorder(variables=[variable])


class Recorder(Behavior):
    """This is the base class to record variables of a network object.
    
    Args:
        variables (list of str): List of variable names to record. A variable should be of tensor type.
        gap_width (int): The intervals of time to record variables. The default is 0.
        tag (str): A tag name for the `Recorder` object. The default is None.
        max_length (int): The history buffer size. If `None`, the variables are recorded \
            for the whole simulation time. The default is None.
        auto_annotate (bool): This parameter specifies whether the variable names include the \
            network object prefix (neurons/synapse/n/s) or not. The default is True.
    """

    initialize_last = True
    visualization_module_outputs = []

    def initialize(self, object):
        super().initialize(object)

        variables = self.parameter("variables", [])
        if variables == []:
            variables = self.parameter("arg_0", [])
        if isinstance(variables, str):
            variables = [variables]

        self.gap_width = self.parameter("gap_width", 0)
        self.max_length = self.parameter("max_length", None)
        self.auto_annotate = self.parameter("auto_annotate", True)
        self.counter = 0
        self.new_data_available = False
        self.variables = {}
        self.compiled = {}

        self.add_variables(variables)
        self.reset()

    def add_variable(self, v):
        self.variables[v] = torch.tensor([], dtype=torch.bool, device=self.device)
        self.compiled[v] = None

    def add_variables(self, vars):
        for v in vars:
            self.add_variable(v)

    def find_objects(self, key):
        result = []
        if key in self.variables:
            result.append(self.variables[key])
        return result

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
        if self.variables[variable].dtype != data.dtype and torch.numel(
            self.variables[variable]
        ):
            print(
                f"WARNING: The recorder received new data with a different datatype({data.dtype}) from the previously recorded data({self.variables[variable].dtype})."
            )
        self.variables[variable] = torch.concat(
            (self.variables[variable], data.unsqueeze(0)), dim=0
        )

    def eq_split(self, eq, splitter):
        str = eq.replace(" ", "")
        parts = []
        str_buf = ""
        for s in str:
            if s in splitter:
                parts.append(str_buf)
                parts.append(s)
                str_buf = ""
            else:
                str_buf += s

        parts.append(str_buf)
        return parts

    def annotate_var_str(self, variable, parent_obj):
        splitter = [
            "*",
            "/",
            "+",
            "-",
            "%",
            ":",
            ";",
            "=",
            "!",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]
        annotated_var = ""
        for part in self.eq_split(variable, splitter):
            if hasattr(parent_obj, part):
                part = "n." + part
            annotated_var += part
        return annotated_var

    def forward(self, parent_obj):
        if parent_obj.recording:
            self.counter += 1

            if self.counter >= self.gap_width:
                self.new_data_available = True
                self.counter = 0

                for v in self.variables:
                    if self.compiled[v] is None:
                        if self.auto_annotate:
                            annotated_var = self.annotate_var_str(v, parent_obj)
                            self.compiled[v] = compile(
                                annotated_var, "<string>", "eval"
                            )
                        else:
                            self.compiled[v] = compile(v, "<string>", "eval")

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
        print("clear")

        for v in self.variables:
            device = self.variables[v].device
            del self.variables[v]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            self.variables[v] = torch.tensor([], dtype=torch.bool, device=device)


class EventRecorder(Recorder):
    """This class is used to record sparse boolean vectors over time more efficiently.
    It returns a tensor of (t, i) tuples where t indicates the time step and i is the 
    index of elements with value `True`. If the variable to record is not boolean itself,
    it is converted to one by assessing whether values are >0.
    
    Args:
        variables (list of str): List of variable names to record.
        gap_width (int): The intervals of time to record variables. The default is 0.
        tag (str): A tag name for the `Recorder` object. The default is None.
        max_length (int): The history buffer size. If `None`, the variables are recorded \
            for the whole simulation time. The default is None.
        auto_annotate (bool): This parameter specifies whether the variable names include the \
            network object prefix (neurons/synapse/n/s) or not. The default is True.
    """

    def find_objects(self, key):
        result = []
        if key in self.variables:
            result.append(self.variables[key])

        if type(key) is str and key[-2:] == ".t" and key[:-2] in self.variables:
            result.append(self.variables[key[:-2]][:, 0])

        if type(key) is str and key[-2:] == ".i" and key[:-2] in self.variables:
            result.append(self.variables[key[:-2]][:, 1])

        return result

    def get_data_v(self, variable, parent_obj):
        n = parent_obj  # used for eval string "n.x"
        s = parent_obj
        neurons = parent_obj
        synapse = parent_obj

        data = eval(self.compiled[variable])

        indices = torch.where(data != 0)
        iteration = torch.ones_like(indices[0]) * parent_obj.iteration

        return torch.stack((iteration, *indices), dim=1)

    def save_data_v(self, data, variable):
        self.variables[variable] = torch.concat([self.variables[variable], data])
