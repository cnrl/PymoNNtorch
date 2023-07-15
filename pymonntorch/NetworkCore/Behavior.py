import torch

from pymonntorch.NetworkCore.Base import TaggableObject
from pymonntorch.utils import is_number


class Behavior(TaggableObject):
    """Base class for behaviors. All behaviors all `TaggableObject`s.

    Attributes:
        tag (str): Tag of the behavior.
        device (str): Device of the behavior. This is overwritten by object's device upon calling `initialize`.
        behavior_enabled (bool): Whether the behavior is enabled. The default is True.
        init_kwargs (dict): Dictionary of the keyword arguments passed to the constructor.
        used_attr_keys (list): List of the name of the attributes that have been used in the `initialize` method.
    """

    initialize_on_init = False
    initialize_last = False

    def __init__(self, *args, **kwargs):
        """Constructor of the `Behavior` class.

        Args:
            **kwargs: Keyword arguments passed to the constructor.
        """
        self.init_kwargs = kwargs
        for i, arg in enumerate(args):
            self.init_kwargs["arg_" + str(i)] = arg
        self.used_attr_keys = []
        self.behavior_enabled = self.parameter("behavior_enabled", True, None)
        super().__init__(
            tag=self.parameter("tag", None, None),
            device=self.parameter("device", None, None),
        )
        self.empty_iteration_function = self.is_empty_iteration_function()
        self.used_attr_keys = torch.nn.ParameterList(self.used_attr_keys)

    def initialize(self, object):
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
        pass

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
                result = object.vector(ds)

            if type(object).__name__ == "SynapseGroup":
                result = object.matrix(ds)

        if plot:
            if type(result) == torch.tensor:
                import matplotlib.pyplot as plt

                plt.hist(result.to("cpu"), bins=30)
                plt.show()

        return result

    def set_parameters_as_variables(self, object):
        """Set the variables defined in the init of behavior as the variables of the object.

        Args:
            object (NetworkObject): The object possessing the behavior.
        """
        for key in self.init_kwargs:
            setattr(object, key, self.parameter(key, None, object=object))
            print("init", key)

    def check_unused_attrs(self):
        """Checks whether all attributes have been used in the `initialize` method."""
        for key in self.init_kwargs:
            if key not in self.used_attr_keys:
                print(
                    'Warning: "'
                    + key
                    + '" not used in initialize of '
                    + str(self)
                    + ' behavior! Make sure that "'
                    + key
                    + '" is spelled correctly and parameter('
                    + key
                    + ",...) is called in initialize. Valid attributes are: "
                    + ", ".join([f'"{param}"' for param in list(self.used_attr_keys)])
                    + "."
                )

    def parameter(
        self,
        key,
        default,
        object=None,
        do_not_diversify=False,
        search_other_behaviors=False,
        tensor=False,
        required=False,
    ):
        """Gets the value of an attribute.

        Args:
            key (str): Name of the attribute.
            default (any): Default value of the attribute.
            object (NetworkObject): The object possessing the behavior.
            do_not_diversify (bool): Whether to diversify the attribute. The default is False.
            search_other_behaviors (bool): Whether to search for the attribute in other behaviors of the object. The default is False.
            tensor (bool): Whether to make a tensor out of value. Suitable for list and numbers.
            required (bool): Whether the attribute is required. The default is False.

        Returns:
            any: The value of the attribute.
        """
        if required and self.init_kwargs.get(key, None) is None:
            print(
                "Warning:",
                key,
                "has to be specified for the behavior with a non None value to run properly.",
                self,
            )

        self.used_attr_keys.append(key)

        result = self.init_kwargs.get(key, default)
        result = default if result is None else result

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

        if tensor and result is not None:
            if object is None:
                raise RuntimeError(
                    f'To turn parameter value of key "{key}" to a tensor, object should not be None.'
                )
            result = torch.tensor(result, device=object.device)
            if result.is_floating_point():
                result = result.to(dtype=object.def_dtype)

        return result

    def is_empty_iteration_function(self):
        """Checks whether a function does anything or not.

        used to stop calling behaviors with empty forward method.
        """
        f = self.forward

        # Returns true if f is an empty function.
        def empty_func():
            pass

        def empty_func_with_docstring():
            """Empty function with docstring."""
            pass

        def constants(f):
            # Return a tuple containing all the constants of a function without: * docstring
            return tuple(x for x in f.__code__.co_consts if x != f.__doc__)

        return (
            f.__code__.co_code == empty_func.__code__.co_code
            and constants(f) == constants(empty_func)
        ) or (
            f.__code__.co_code == empty_func_with_docstring.__code__.co_code
            and constants(f) == constants(empty_func_with_docstring)
        )
