import torch

from pymonntorch.NetworkCore.TaggableObject import *


class NetworkObject(TaggableObject):
    """This is the base class for all network objects.

    This class is used to treat network objects' behaviors and is a subclass of TaggableObject.

    Attributes:
        network (Network): The parent network object.
        behavior (list or dict): List or dictionary of behaviors.
        analysis_modules (list): List of analysis modules.
    """

    def __init__(self, tag, network, behavior, device="cpu"):
        """Initialize the object.

        Args:
            tag (str): Tag to add to the object. It can also be a comma-separated string of multiple tags.
            network (Network): The parent network object.
            behavior (list or dict): List or dictionary of behaviors. If a dictionary is used, the keys must be integers.
            device (str): Device on which the object is located. The default is "cpu".
        """
        super().__init__(tag, device)

        self.network = network
        self.behavior = behavior
        if type(behavior) == list:
            self.behavior = dict(zip(range(len(behavior)), behavior))
        # self.behavior = torch.nn.ModuleDict(self.behavior)

        for k in sorted(list(self.behavior.keys())):
            if self.behavior[k].set_variables_on_init:
                network._set_variables_check(self, k)

        self.analysis_modules = []

    def register_behavior(self, key, behavior, initialize=True):
        """Register a single behavior to the network object.

        Args:
            key (str): Key to be used to access behavior.
            behavior (Behavior): Behavior to be registered.
            initialize (bool): If true, behavior will be initialized. The default is True.

        Returns:
            Behavior: The behavior.
        """
        self.behavior[key] = behavior
        self.network._add_key_to_sorted_behavior_timesteps(key)
        self.network.clear_tag_cache()

        if initialize:
            behavior.set_variables(self)
            behavior.check_unused_attrs()

        return behavior

    def register_behaviors(self, behavior_dict):
        """Register multiple behaviors to the network object.

        Args:
            behavior_dict (dict): Dictionary of behaviors to be registered. The keys must be integers.

        Returns:
            dict: The dictionary of behaviors.
        """
        for key, behavior in behavior_dict.items():
            self.register_behavior(key, behavior)
        return behavior_dict

    def remove_behavior(self, key_tag_behavior_or_type):
        """Remove behavior(s) from the network object.

        Args:
            key_tag_behavior_or_type (str, Behavior, or type): Key, tag, behavior object, or type of behavior to be removed.
        """
        remove_keys = []
        for key in self.behavior:
            b = self.behavior[key]
            if (
                key_tag_behavior_or_type == key
                or key_tag_behavior_or_type in b.tags
                or key_tag_behavior_or_type == b
                or key_tag_behavior_or_type == type(b)
            ):
                remove_keys.append(key)

        for key in remove_keys:
            self.behavior.pop(key)

    def set_behaviors(self, tag, enabled):
        """Set behaviors to be enabled or disabled.

        Args:
            tag (str): Tag of behaviors to be enabled or disabled.
            enabled (bool): If true, behaviors will be enabled. If false, behaviors will be disabled.
        """
        if enabled:
            print("activating", tag)
        else:
            print("deactivating", tag)
        for b in self[tag]:
            b.behavior_enabled = enabled

    def deactivate_behaviors(self, tag):
        """Disable behaviors.

        Args:
            tag (str): Tag of behaviors to be disabled.
        """
        self.set_behaviors(tag, False)

    def activate_behaviors(self, tag):
        """Enable behaviors.

        Args:
            tag (str): Tag of behaviors to be enabled.
        """
        self.set_behaviors(tag, True)

    def find_objects(self, key):
        """Find behaviors and analysis modules in the network object by key.

        Args:
            key (str): Key to be used to access behavior or analysis module.

        Returns:
            list: List of behaviors and analysis modules.
        """
        result = []

        if key in self.behavior:
            result.append(self.behavior[key])

        for bk in self.behavior:
            behavior = self.behavior[bk]
            result += behavior[key]

        for am in self.analysis_modules:
            result += am[key]

        return result

    def register_analysis_module(self, module):
        """Register an analysis module to the network object.

        Args:
            module (AnalysisModule): Analysis module to be registered.
        """
        module._attach_and_initialize_(self)

    def get_all_analysis_module_results(self, tag, return_modules=False):
        """Get results from all analysis modules in the network object.

        Args:
            tag (str): Tag of analysis modules to be used.
            return_modules (bool): If true, the analysis modules will be returned. The default is False.

        Returns:
            dict: Dictionary of results.
        """
        result = {}
        modules = {}
        for module in self[tag]:
            module_results = module.get_results()
            for k in module_results:
                result[k] = module_results[k]
                modules[k] = module

        if return_modules:
            return result, modules
        else:
            return result

    def buffer_roll(self, mat, new=None):
        """Shift the elements of a tensor to the right.

        Args:
            mat (torch.Tensor): Tensor to be shifted.
            new (int or float or bool or torch.Tensor): New element to be inserted at the beginning of the tensor. The default is None (i.e. Nothing is added).

        Returns:
            torch.Tensor: The shifted tensor.
        """
        mat[1 : len(mat)] = mat[0 : len(mat) - 1]

        if new is not None:
            mat[0] = new

        return mat

    def _get_mat(self, mode, dim, scale=None, density=None, plot=False, kwargs={}):
        """Get a tensor with object's dimensionality.

        The tensor can be initialized in different modes. List of possible values for mode includes:
        - "random" or "rand" or "rnd" or "uniform": Uniformly distributed random numbers in range [0, 1).
        - "normal": Normally distributed random numbers with zero mean and unit variance.
        - "ones": Tensor filled with ones.
        - "zeros": Tensor filled with zeros.
        - A single number: Tensor filled with that number.
        - You can also use any function from torch package for this purpose. Note that you should **not** use `torch.` prefix.

        Args:
            mode (str): Mode to be used to initialize tensor.
            dim (int or tuple of int): Dimensionality of the tensor.
            scale (float): Scale of the tensor. The default is None (i.e. No scaling is applied).
            density (float): Density of the tensor. The default is None (i.e. dense tensor).
            plot (bool): If true, the histogram of the tensor will be plotted. The default is False.
            kwargs (dict): Keyword arguments to be passed to the initialization function.

        Returns:
            torch.Tensor: The initialized tensor."""
        prefix = "torch."
        if mode == "random" or mode == "rand" or mode == "rnd" or mode == "uniform":
            mode = "rand"

        if type(mode) == int or type(mode) == float:
            mode = "ones()*" + str(mode)

        mode = prefix + mode
        if "(" not in mode and ")" not in mode:
            mode += "()"

        if "device" in kwargs:
            kwargs.pop("device")

        if mode not in self._mat_eval_dict:
            a1 = "dim,device=" + f"'{self.device}'"
            if "()" in mode:  # no arguments => no comma
                ev_str = mode.replace(")", a1 + ",**kwargs)")
            else:
                ev_str = mode.replace(")", "," + a1 + ",**kwargs)")

            self._mat_eval_dict[mode] = compile(ev_str, "<string>", "eval")

        result = eval(self._mat_eval_dict[mode])

        if density is not None:
            if type(density) == int or type(density) == float:
                result = result * (torch.rand(dim, device=self.device) <= density)
            elif type(density) is torch.tensor:
                result = result * (
                    torch.rand(dim, device=self.device) <= density[:, None]
                )

        if scale is not None:
            result *= scale

        if plot:
            import matplotlib.pyplot as plt

            plt.hist(result.flatten().to("cpu"), bins=30)
            plt.show()

        if "dtype" in kwargs:
            return result
        return result.to(def_dtype)

    def get_buffer_mat(self, dim, size, **kwargs):
        """Get a buffer of specific size with object's dimensionality.

        Args:
            dim (int or tuple of int): Dimensionality of the buffer.
            size (int): Size of the buffer.
            kwargs (dict): Keyword arguments to be passed to the initialization function.

        Returns:
            torch.Tensor: The buffer.
        """
        return (
            torch.cat([torch.zeros(dim, **kwargs) for _ in range(size)])
            .reshape(size, *dim)
            .to(self.device)
        )

    @property
    def iteration(self):
        """int: iteration number or time step."""
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        if iteration >= 0 and type(iteration) is int:
            self._iteration = iteration
        else:
            print(
                "WARNING: Attempting to set an invalid value for iteration!\n Setting iteration to zero..."
            )
            self._iteration = 0
