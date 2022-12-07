import torch

from pymonntorch.NetworkCore.Base import TaggableObject
from pymonntorch.utils import is_number


class Behavior(TaggableObject):
    set_variables_on_init = False

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.used_attr_keys = []
        self.behavior_enabled = self.get_init_attr("behavior_enabled", True, None)
        super().__init__(
            tag=self.get_init_attr("tag", None, None),
            device=self.get_init_attr("device", None, None),
        )
        self.used_attr_keys = torch.nn.ParameterList(self.used_attr_keys)

    def set_variables(self, object):
        self.device = object.device
        return

    def new_iteration(self, object):
        return

    def __repr__(self):
        result = self.__class__.__name__ + "("
        for k in self.init_kwargs:
            result += str(k) + "=" + str(self.init_kwargs[k]) + ","
        result += ")"
        return result

    def evaluate_diversity_string(self, ds, neurons_or_synapses):
        if "same(" in ds and ds[-1] == ")":
            params = ds[5:-1].replace(" ", "").split(",")
            if len(params) == 2:
                return getattr(neurons_or_synapses[params[0], 0], params[1])

        plot = False
        if ";plot" in ds:
            ds = ds.replace(";plot", "")
            plot = True

        result = ds

        if "(" in ds and ")" in ds:  # is function
            if type(neurons_or_synapses).__name__ == "NeuronGroup":
                result = neurons_or_synapses.get_neuron_vec(ds)

            if type(neurons_or_synapses).__name__ == "SynapseGroup":
                result = neurons_or_synapses.get_synapse_mat(ds)

        if plot:
            if type(result) == torch.tensor:
                import matplotlib.pyplot as plt

                plt.hist(result.to("cpu"), bins=30)
                plt.show()

        return result

    def set_init_attrs_as_variables(self, object):
        for key in self.init_kwargs:
            setattr(object, key, self.get_init_attr(key, None, object=object))
            print("init", key)

    def check_unused_attrs(self):
        for key in self.init_kwargs:
            if not key in self.used_attr_keys:
                print(
                    'Warning: "'
                    + key
                    + '" not used in set_variables of '
                    + str(self)
                    + ' behavior! Make sure that "'
                    + key
                    + '" is spelled correctly and get_init_attr('
                    + key
                    + ",...) is called in set_variables. Valid attributes are:"
                    + str(self.used_attr_keys)
                )

    def get_init_attr(
        self,
        key,
        default,
        object=None,
        do_not_diversify=False,
        search_other_behaviors=False,
        required=False,
    ):
        if required and not key in self.init_kwargs:
            print(
                "Warning:",
                key,
                "has to be specified for the behavior to run properly.",
                self,
            )

        self.used_attr_keys.append(key)

        result = self.init_kwargs.get(key, default)

        if (
            key not in self.init_kwargs
            and object is not None
            and search_other_behaviors
        ):
            for b in object.behaviors:
                if key in b.init_kwargs:
                    result = b.init_kwargs.get(key, result)

        if not do_not_diversify and type(result) is str and object is not None:
            result = self.evaluate_diversity_string(result, object)

        if type(result) is str and default is not None:
            if "%" in result and is_number(result.replace("%", "")):
                result = str(float(result.replace("%", "")) / 100.0)

            result = type(default)(result)

        return result
