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
        self.behavior = behavior if behavior is not None else {}
        if type(behavior) == list:
            self.behavior = dict(zip(range(len(behavior)), behavior))
        # self.behavior = torch.nn.ModuleDict(self.behavior)

        for b in self.behavior.values():
            if not hasattr(self, b.tags[0]):
                setattr(self, b.tags[0], b)

        for k, b in self.behavior.items():
            self.network._add_behavior_to_sorted_execution_list(k, self, b)

        for k in sorted(list(self.behavior.keys())):
            if self.behavior[k].initialize_on_init:
                self.behavior[k].initialize(self)

        self.analysis_modules = []

        self.recording = True

    def add_behavior(self, key, behavior, initialize=True):
        """Add a single behavior to the network object.

        Args:
            key (str): Key to be used to access behavior.
            behavior (Behavior): Behavior to be added.
            initialize (bool): If true, behavior will be initialized. The default is True.

        Returns:
            Behavior: The behavior.
        """
        if key not in self.behavior:
            self.behavior[key] = behavior
            self.network._add_behavior_to_sorted_execution_list(
                key, self, self.behavior[key]
            )
            self.network.clear_tag_cache()
            if initialize:
                behavior.initialize(self)
                behavior.check_unused_attrs()
            return behavior
        else:
            raise Exception("Error: Key already exists." + str(key))

    def add_behaviors(self, behavior_dict):
        """Add multiple behaviors to the network object.

        Args:
            behavior_dict (dict): Dictionary of behaviors to be added. The keys must be integers.

        Returns:
            dict: The dictionary of behaviors.
        """
        for key, behavior in behavior_dict.items():
            self.add_behavior(key, behavior)
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
            b = self.behavior.pop(key)
            self.network._remove_behavior_from_sorted_execution_list(key, self, b)

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

    def add_analysis_module(self, module):
        """Add an analysis module to the network object.

        Args:
            module (AnalysisModule): Analysis module to be added.
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

    def buffer_roll(self, mat, new=None, counter=False):
        """Shift the elements of a tensor to the right.

        Args:
            mat (torch.Tensor): Tensor to be shifted.
            new (int or float or bool or torch.Tensor): New element to be inserted at the beginning of the tensor. The default is None (i.e. the last of the buffer is repositoined at the first).
            counter (bool): If True, rolling is done in opposite direction, and the new element is added at the end.

        Returns:
            torch.Tensor: The shifted tensor.
        """
        mat = mat.roll(1 - (2 * counter), dims=0)

        if new is not None:
            mat[0 - counter] = new

        return mat

    def _get_mat(self, mode, dim, scale=None, density=None, plot=False, dtype=None):
        """Get a tensor with object's dimensionality.

        The tensor can be initialized in different modes. List of possible values for mode includes:
        - "random" or "rand" or "rnd" or "uniform": Uniformly distributed random numbers in range [0, 1).
        - "normal(mean=a, std=b)": Normally distributed random numbers with `a` as mean and `b` as standard derivation.
        - "ones": Tensor filled with ones.
        - "zeros": Tensor filled with zeros.
        - A single number: Tensor filled with that number.
        - You can also use any function from torch package for this purpose.

        Args:
            mode (str): Mode to be used to initialize tensor.
            dim (int or tuple of int): Dimensionality of the tensor.
            scale (float): Scale of the tensor. The default is None (i.e. No scaling is applied).
            density (float): Density of the tensor. The default is None (i.e. dense tensor).
            plot (bool): If true, the histogram of the tensor will be plotted. The default is False.
            dtype (str or type): Data type of the tensor. If None, `def_dtype` will be used.

        Returns:
            torch.Tensor: The initialized tensor."""

        dtype = self.def_dtype if dtype is None else dtype

        if mode not in self._mat_eval_dict:
            prefix = "torch."
            ev_str = mode

            if (
                ev_str == "random"
                or ev_str == "rand"
                or ev_str == "rnd"
                or ev_str == "uniform"
            ):
                ev_str = "rand"

            if type(ev_str) == int or type(ev_str) == float:
                ev_str = "ones()*" + str(ev_str)

            ev_str = prefix + ev_str
            if "(" not in ev_str and ")" not in ev_str:
                ev_str += "()"

            a1 = "size=dim,device=self.device,dtype=dtype)"

            # check for positional argument
            if ev_str[ev_str.index("(") + 1 : ev_str.index(")")].strip():
                a1 = "," + a1

            ev_str = ev_str.replace(")", a1)
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

        return result

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
        return self.network._iteration

    @iteration.setter
    def iteration(self, iteration):
        if iteration >= 0 and type(iteration) is int:
            self.network._iteration = iteration
        else:
            print(
                "WARNING: Attempting to set an invalid value for iteration!\n Setting iteration to zero..."
            )
            self.network._iteration = 0

    @property
    def def_dtype(self):
        return self.network._def_dtype

    @def_dtype.setter
    def def_dtype(self, dtype):
        self.network._def_dtype = dtype
