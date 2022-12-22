import inspect

from pymonntorch.NetworkCore.Base import *

# This class can be used to add tag-searchable functions to the neurongroups, synapsegroups and the network object.
# It has a main execute function which can be called with module(...) or module.exec(...)
# Other "normal" functions can be added as well.
# Via the add_tag function, the modules can be categorized into groups


class AnalysisModule(TaggableObject):
    """This class can be used to add tag-searchable functions to the neurongroups, synapsegroups and the network object.
    
    It has a main execute function which can be called with module(...) or module.exec(...).
    Other "normal" functions can be added as well. Via the add_tag function, the modules can be categorized into groups.
    
    Attributes:
        parent (NeuronGroup, SynapseGroup or Network): The parent object.
        init_kwargs (dict): The arguments passed to the constructor.
        used_attr_keys (list): The list of the used attributes.
        execution_arguments (dict): The arguments of the execute function.
        result_storage (dict): The dictionary of argument results after the execute function.
        current_result (any): The result of the last execute function call.
        save_results (bool): If the results should be saved.
        update_notifier_functions (list): The list of the update notifier functions.
        progress_update_function (function): The function to update the progress.
    """
    def __init__(self, parent=None, **kwargs):
        self.init_kwargs = kwargs
        super().__init__(tag=self.get_init_attr("tag", None))

        self.used_attr_keys = []

        self.add_tag(self.__class__.__name__)

        self.execution_arguments = self._get_default_args_(self.execute)

        self.result_storage = {}  # key=arguments value=result from run
        self.current_result = None
        self.save_results = True

        self.update_notifier_functions = []

        self.progress_update_function = None

        if parent is not None:
            self._attach_and_initialize_(parent)

    def add_progress_update_function(self, function):
        """Adds a function to update the progress of the module.
        
        Args:
            function (function): The function to update the progress.
        """
        self.progress_update_function = function

    def update_progress(self, percent):
        """Updates the progress of the module.

        Args:
            percent (int): The progress in percent.
        """
        if self.progress_update_function is not None:
            self.progress_update_function(percent)

    def _attach_and_initialize_(self, parent):
        """Attaches the module to the parent object and initializes it.
        
        Args:
            parent (NeuronGroup, SynapseGroup or Network): The parent object.
        """
        self.parent = parent
        parent.analysis_modules.append(self)
        setattr(parent, self.__class__.__name__, self)
        self.initialize(parent)

    def initialize(self, object):
        """This function is called when the module is attached to the parent object. It should be overridden.

        - access arguments via self.get_init_attr(key, default)
        - add tag via self.add_tag(tag)
        - add execution arguments via self.add_execution_argument(...)
        `execute` does not have to be used.
        
        Args:
            object (NeuronGroup, SynapseGroup or Network): The parent object.
        """
        return

    def execute(
        self, object, **kwargs
    ):  
        """Executes the functions of the module with the given arguments. It should be overridden.
        
        Note: Do not call this function directly. Use the instance(...) instead of instance.execute(...).

        Args:
            object (NeuronGroup, SynapseGroup or Network): The parent object.
            **kwargs: The arguments of the function.
        """
        return

    def is_executable(self):
        return type(self).execute != AnalysisModule.execute

    def get_init_attr(self, key, default):
        """Returns the value of the given key from the init arguments. If the key is not present, the default value is returned.
        
        Args:
            key (str): The name of the argument.
            default (any): The default value.

        Returns:
            any: The value of the argument.
        """
        if key in self.init_kwargs:
            return self.init_kwargs[key]
        else:
            return default

    def _update_notification_(self, key=None):
        for function in self.update_notifier_functions:
            function(key)

    def remove_update_notifier(self, function):
        """Removes the given function from the update notifier functions.
        
        Args:
            function (function): The function to remove.
        """
        if function in self.update_notifier_functions:
            self.update_notifier_functions.remove(function)

    def set_update_notifier(self, function):
        """Adds the given function ro the list of update notifier functions.
        
        Args:
            function (function): The function to set.
        """
        self.update_notifier_functions.append(function)

    def __call__(self, **kwargs):
        self.update_progress(0)
        self.current_key = self.generate_current_key(kwargs)
        return self.save_result(self.current_key, self.execute(self.parent, **kwargs))

    def exec(self, **kwargs):
        """Executes the module with the given arguments.
        
        Args:
            **kwargs: The arguments of the function.
        
        Returns:
            any: The result of the function.
        """
        self.update_progress(0)
        self.current_key = self.generate_current_key(kwargs)
        return self.save_result(self.current_key, self.execute(self.parent, **kwargs))

    def _get_base_name_(self):
        return self.__class__.__name__

    def generate_current_key(self, args_key, add_args=True):
        """Generates the key for the current execution.
        
        Args:
            args_key (list): The arguments of the function.
            add_args (bool): If the arguments should be added to the key.
        """
        key = self._get_base_name_()
        if len(args_key) > 0 and add_args:
            key += " " + str(args_key)
        return key

    def save_result(self, key, result):
        """Saves the result of the execution.
        
        Args:
            key (str): The key of the execution.
            result (any): The result of the execution.
        
        Returns:
            any: The result of the execution.
        """
        self.update_progress(100)
        if self.save_results and result is not None:
            self.current_result = result
            self.result_storage[key] = result
        self._update_notification_(key)
        return result

    def last_call_result(self):
        """Returns the result of the last execution."""
        return self.current_result

    def get_results(self):
        """Returns the results of all executions."""
        return self.result_storage

    def remove_result(self, key):
        """Removes the result of the given key from the dictionary of results.
        
        Args:
            key (str): The key of the execution to remove.
        """
        if key in self.result_storage:
            return self.result_storage.pop(key)
        else:
            print("cannot remove result", key, "not found.")
        self._update_notification_()

    def clear_results(self):
        """Removes all results."""
        self.result_storage = {}
        self._update_notification_()

    def _get_default_args_(
        self, func, exclude=["self", "args", "kwargs"], exclude_first=True
    ):
        result = {}
        signature = inspect.signature(func)
        i = 0
        for k, v in signature.parameters.items():
            if i > 0 or not exclude_first:
                if k not in exclude:
                    if v.default is not inspect.Parameter.empty:
                        result[k] = v.default
                    else:
                        result[k] = ""
            i += 1
        return result
