import torch

from pymonntorch.NetworkCore.Base import TaggableObject
from pymonntorch.utils import is_number


class Behavior(TaggableObject):
    """Base class for behaviors. All behaviors all `TaggableObject`s.
    
    Attributes:
        tag (str): Tag of the behavior.
        device (str): Device of the behavior. This is overwritten by object's device upon calling `set_variables`.
        behavior_enabled (bool): Whether the behavior is enabled. The default is True.
        init_kwargs (dict): Dictionary of the keyword arguments passed to the constructor.
        used_attr_keys (list): List of the name of the attributes that have been used in the `set_variables` method.
    """
    set_variables_on_init = False

    def __init__(self, **kwargs):
        """Constructor of the `Behavior` class.
        
        Args:
            **kwargs: Keyword arguments passed to the constructor.
        """
        self.init_kwargs = kwargs
        self.used_attr_keys = []
        self.behavior_enabled = self.get_init_attr("behavior_enabled", True, None)
        super().__init__(
            tag=self.get_init_attr("tag", None, None),
            device=self.get_init_attr("device", None, None),
        )
        self.used_attr_keys = torch.nn.ParameterList(self.used_attr_keys)

    def set_variables(self, object):
        """Sets the variables of the object. This method is called by the `Network` class when the object is added to the network.
        
        **Note:** All sub-classes of `Behavior` overriding this method should call the super method to ensure everything is placed on the correct device.
        
        Args:
            object (TaggableObject): Object possessing the behavior.
        """
        self.device = object.device
        return

    def forward(self, object):
        """Forward pass of the behavior. This method is called by the `Network` class per simulation iteration.
        
        Args:
            object (TaggableObject): Object possessing the behavior.
        """
        return

    def __repr__(self):
        result = self.__class__.__name__ + "("
        for k in self.init_kwargs:
            result += str(k) + "=" + str(self.init_kwargs[k]) + ","
        result += ")"
        return result

    def evaluate_diversity_string(self, ds, object):
        """Evaluates the diversity string describing tensors of an object.
        
        Args:
            ds (str): Diversity string describing the tensors of the object.
            object (NetworkObject): The object possessing the behavior.

        Returns:
            torch.tensor: The resulting tensor.
        """
        if "same(" in ds and ds[-1] == ")":
            params = ds[5:-1].replace(" ", "").split(",")
            if len(params) == 2:
                return getattr(object[params[0], 0], params[1])

        plot = False
        if ";plot" in ds:
            ds = ds.replace(";plot", "")
            plot = True

        result = ds

        if "(" in ds and ")" in ds:  # is function
            if type(object).__name__ == "NeuronGroup":
                result = object.get_neuron_vec(ds)

            if type(object).__name__ == "SynapseGroup":
                result = object.get_synapse_mat(ds)

        if plot:
            if type(result) == torch.tensor:
                import matplotlib.pyplot as plt

                plt.hist(result.to("cpu"), bins=30)
                plt.show()

        return result

    def set_init_attrs_as_variables(self, object):
        """Set the variables defined in the init of behavior as the variables of the object.
        
        Args:
            object (NetworkObject): The object possessing the behavior.
        """
        for key in self.init_kwargs:
            setattr(object, key, self.get_init_attr(key, None, object=object))
            print("init", key)

    def check_unused_attrs(self):
        """Checks whether all attributes have been used in the `set_variables` method."""
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
        """Gets the value of an attribute.
        
        Args:
            key (str): Name of the attribute.
            default (any): Default value of the attribute.
            object (NetworkObject): The object possessing the behavior.
            do_not_diversify (bool): Whether to diversify the attribute. The default is False.
            search_other_behaviors (bool): Whether to search for the attribute in other behaviors of the object. The default is False.
            required (bool): Whether the attribute is required. The default is False.

        Returns:
            any: The value of the attribute.
        """
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
