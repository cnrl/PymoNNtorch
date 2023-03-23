import time

import torch

from pymonntorch.NetworkCore.Base import *
from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.NetworkCore.SynapseGroup import *


class Network(NetworkObject):
    """This is the class to construct a neural network.

    This is the placeholder of all neural network components to be simulated.
    All objects will receive an instance of this class.

    Attributes:
        NeuronGroups (list): List of all NeuronGroups in the network.
        SynapseGroups (list): List of all SynapseGroups in the network.
        behavior (list or dict): List of all network-specific behaviors.
        device (str): Device to run the simulation on. Either 'cpu' or 'cuda'.
    """
    def __init__(self, tag=None, behavior={}, device="cpu"):
        """Initialize the network.

        Args:
            tag (str): Tag to add to the network. It can also be a comma-separated string of multiple tags.
            behavior (list or dict): List or dictionary of behaviors. If a dictionary is used, the keys must be integers.
            device (str): Device on which the network is located. The default is "cpu".
        """
        super().__init__(tag, self, behavior, device=device)

        self.NeuronGroups = []
        self.SynapseGroups = []

        self._iteration = 0

    def set_behaviors(self, tag, enabled):
        """Set behaviors of specific tag to be enabled or disabled.
        
        Args:
            tag (str): Tag of behaviors to be enabled or disabled.
            enabled (bool): If true, behaviors will be enabled. If false, behaviors will be disabled.
        """
        if enabled:
            print("activating", tag)
        else:
            print("deactivating", tag)
        for obj in self.all_objects():
            for b in obj[tag]:
                b.behavior_enabled = enabled

    def recording_off(self):
        """Turn off recording for all objects in the network."""
        for obj in self.all_objects():
            obj.recording = False

    def recording_on(self):
        """Turn on recording for all objects in the network."""
        for obj in self.all_objects():
            obj.recording = True

    def all_objects(self):
        """Return a list of all objects in the network."""
        l = [self]
        l.extend(self.NeuronGroups)
        l.extend(self.SynapseGroups)
        return l

    def all_behaviors(self):
        """Return a list of all behaviors in the network."""
        result = []
        for obj in self.all_objects():
            for beh in obj.behavior.values():
                result.append(beh)
        return result

    def clear_recorder(self, keys=None):
        """Clear the recorder objects of all network components."""
        for obj in self.all_objects():
            for key in obj.behavior:
                if (keys is None or key in keys) and hasattr(
                    obj.behavior[key], "clear_recorder"
                ):
                    obj.behavior[key].clear_recorder()

    def __repr__(self):
        neuron_count = torch.sum(torch.tensor([ng.size for ng in self.NeuronGroups]))
        synapse_count = torch.sum(
            torch.tensor([sg.src.size * sg.dst.size for sg in self.SynapseGroups])
        )

        basic_info = (
            "(Neurons: "
            + str(neuron_count)
            + "|"
            + str(len(self.NeuronGroups))
            + " groups, Synapses: "
            + str(synapse_count)
            + "|"
            + str(len(self.SynapseGroups))
            + " groups)"
        )

        result = "Network" + str(self.tags) + basic_info + "{"
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        result += "}" + "\r\n"

        for ng in self.NeuronGroups:
            result += str(ng) + "\r\n"

        used_tags = []
        for sg in self.SynapseGroups:
            tags = str(sg.tags)
            if tags not in used_tags:
                result += str(sg) + "\r\n"
            used_tags.append(tags)

        return result[:-2]

    def find_objects(self, key):
        """Find objects in the network with a specific tag.
        
        Args:
            key (str): Tag to search for.

        Returns:
            list: List of objects with the tag.
        """
        result = super().find_objects(key)

        for ng in self.NeuronGroups:
            result.extend(ng[key])

        for sg in self.SynapseGroups:
            result.extend(sg[key])

        for am in self.analysis_modules:
            result.extend(am[key])

        return result

    def initialize(self, info=True, warnings=True, storage_manager=None):
        """Initialize the variables of the network and all its components.
        
        Args:
            info (bool): If true, print information about the network.
            warnings (bool): If true, print warnings while checking the tag uniqueness.
            storage_manager (StorageManager): Storage manager to use for the network.
        """
        if info:
            desc = str(self)
            print(desc)
            if storage_manager is not None:
                storage_manager.save_param("info", desc)

        self.set_synapses_to_neuron_groups()
        self.behavior_timesteps = []

        for obj in self.all_objects():
            for ind in obj.behavior:
                self._add_key_to_behavior_timesteps(ind)

        self.set_variables()

        self.check_unique_tags(warnings)

    def _add_key_to_behavior_timesteps(self, ind):
        if ind not in self.behavior_timesteps:
            self.behavior_timesteps.append(ind)
            self.behavior_timesteps.sort()

    def check_unique_tags(self, warnings=True):
        """Check if all tags in the network are unique. In case of doubles, a new tag will be 
        automatically assigned to second instance.
        
        Args:
            warnings (bool): Whether to log the warnings or not.
        """
        unique_tags = []
        for ng in self.NeuronGroups:

            if len(ng.tags) == 0:
                ng.tags.append("NG")
                print('no tag defined for NeuronGroup. "NG" tag added')

            if ng.tags[0] in unique_tags:
                counts = unique_tags.count(ng.tags[0])
                new_tag = ng.tags[0] + chr(97 + counts)
                unique_tags.append(ng.tags[0])
                if warnings:
                    print(
                        'Warning: NeuronGroup Tag "'
                        + ng.tags[0]
                        + '" already in use. The first Tag of an Object should be unique and will be renamed to "'
                        + new_tag
                        + '". Multiple Tags can be separated with a "," (NeuronGroup(..., tag="tag1,tag2,..."))'
                    )
                ng.tags[0] = new_tag

            else:
                unique_tags.append(ng.tags[0])

    def clear_tag_cache(self):
        """Clear the tag cache of all objects in the network for faster search."""
        for obj in self.all_objects():
            obj.clear_cache()

            for k in obj.behavior:
                obj.behavior[k].clear_cache()

    def _set_variables_check(self, obj, key):
        obj_keys_before = list(obj.__dict__.keys())
        beh_keys_before = list(obj.behavior[key].__dict__.keys())
        obj.behavior[key].set_variables(obj)
        obj_keys_after = list(obj.__dict__.keys())
        beh_keys_after = list(obj.behavior[key].__dict__.keys())
        obj.behavior[key]._created_obj_variables = list(
            set(obj_keys_after) - set(obj_keys_before)
        )
        obj.behavior[key]._created_beh_variables = list(
            set(beh_keys_after) - set(beh_keys_before)
        )

    def set_variables(self):
        """Set the variables of all objects' behaviors in the network."""
        for timestep in self.behavior_timesteps:

            for obj in self.all_objects():

                if timestep in obj.behavior:
                    if not obj.behavior[timestep].set_variables_on_init:
                        self._set_variables_check(obj, timestep)
                        obj.behavior[timestep].check_unused_attrs()

    def set_synapses_to_neuron_groups(self):
        """Set the synapses of all synapse groups to the corresponding neuron groups."""
        for ng in self.NeuronGroups:

            ng.afferent_synapses = {"All": []}
            ng.efferent_synapses = {"All": []}

            for sg in self.SynapseGroups:
                for tag in sg.tags:
                    ng.afferent_synapses[tag] = []
                    ng.efferent_synapses[tag] = []

            for sg in self.SynapseGroups:
                if sg.dst.BaseNeuronGroup == ng:
                    for tag in sg.tags + ["All"]:
                        ng.afferent_synapses[tag].append(sg)

                if sg.src.BaseNeuronGroup == ng:
                    for tag in sg.tags + ["All"]:
                        ng.efferent_synapses[tag].append(sg)

    def simulate_iteration(self, measure_behavior_execution_time=False):
        """Simulate one iteration of the network.

        Each iteration includes a `forward` call of objects' behaviors in the order of their keys in the dictionary or list index.

        Args:
            measure_behavior_execution_time (bool): Whether to measure the actual execution time of the behaviors.

        Returns:
            None or dict: If `measure_behavior_execution_time` is set to True, a dictionary with the execution times of the behaviors is returned.
        """
        if measure_behavior_execution_time:
            time_measures = {timestep:0.0 for timestep in self.behavior_timesteps}

        self.iteration += 1

        for timestep in self.behavior_timesteps:
            objects = self.all_objects()
            for obj in objects:
                obj.iteration = self.iteration
                if timestep in obj.behavior and obj.behavior[timestep].behavior_enabled:
                    if measure_behavior_execution_time:
                        start_time = time.time()
                        obj.behavior[timestep](obj)
                        time_measures[timestep] += (time.time() - start_time) * 1000
                    else:
                        obj.behavior[timestep](obj)

        if measure_behavior_execution_time:
            return time_measures

    def simulate_iterations(
        self,
        iterations,
        batch_size=-1,
        measure_block_time=True,
        disable_recording=False,
        batch_progress_update_func=None,
    ):
        """Simulates the network for a number of iterations.
        
        Args:
            iterations (int): Number of iterations to simulate.
            batch_size (int): Number of iterations to simulate in one batch. If set to -1, the whole simulation is done in one batch.
            measure_block_time (bool): Whether to measure the time of each batch.
            disable_recording (bool): Whether to disable the recording of the network.
            batch_progress_update_func (function): Function to call after each batch. The function should take the current batch number and network instance as arguments.
        """
        if type(iterations) is str:
            iterations = self["Clock", 0].time_to_iterations(iterations)

        time_diff = None

        if disable_recording:
            self.recording_off()

        if batch_size == -1:
            outside_it = 1
            block_iterations = iterations
        else:
            outside_it = int(iterations / batch_size)
            block_iterations = batch_size

        for t in range(int(outside_it)):
            if measure_block_time:
                start_time = time.time()
            for i in range(int(block_iterations)):
                self.simulate_iteration()
            if measure_block_time:
                time_diff = (time.time() - start_time) * 1000

                print(
                    "\r{}xBatch: {}/{} ({}%) {:.3f}ms".format(
                        block_iterations,
                        t + 1,
                        outside_it,
                        int(100 / outside_it * (t + 1)),
                        time_diff,
                    ),
                    end="",
                )

            if batch_progress_update_func is not None:
                batch_progress_update_func((t + 1.0) / int(outside_it) * 100.0, self)

        for _ in range(iterations % batch_size):
            self.simulate_iteration()

        if disable_recording:
            self.recording_on()

        if measure_block_time:
            print("")

        return time_diff
