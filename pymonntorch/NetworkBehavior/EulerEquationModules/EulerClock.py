from pymonntorch.NetworkCore.Behavior import Behavior

from sympy.physics.units import convert_to, second, seconds


class Clock(Behavior):
    def initialize(self, neuron_or_network):
        super().initialize(neuron_or_network)

        self.add_tag("Clock")

        neuron_or_network.clock_step_size = float(
            convert_to(eval(self.parameter("step", "1*ms")), seconds) / second
        )  # in ms (*ms)
        self.clock_step_size = neuron_or_network.clock_step_size
        print(neuron_or_network.clock_step_size)
        neuron_or_network.t = 0.0

    def forward(self, neuron_or_network):
        neuron_or_network.t += neuron_or_network.clock_step_size

    def time_to_iterations(self, time_str):
        iterations = int(
            convert_to(eval(time_str + "/seconds"), seconds) / self.clock_step_size
        )
        return iterations
