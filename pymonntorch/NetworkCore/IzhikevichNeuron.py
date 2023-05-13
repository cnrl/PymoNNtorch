# -*- encoding: utf-8 -*-
# __author__ = "Keywan Tajfar"

from pymonntorch import *

#==============================================================================#

class Izhikevich_Neuron(Behavior):
	r"""
	Implementing the Izhikevich Model dynamics.
	by Izhikevich, 2003
	the formula :
				dv/dt = 0.04v^2 + 5v + 140 - u + I
				du/dt = a(bv - u)
				if v > 30 mV then {v = c and u = u + d}
	"""
 
	def initialize(self, neurons, c=-65.0, u=-14.0, a=0.02, b=0.2, d=8.0, threshold=30.0, dt=0.5, voltage_i=0.0):
		super().initialize(neurons)

		# Internal state of neuron
		neurons.voltage = neurons.vector(mode = "uniform") + c 			# Tensor of membrane potentials
		neurons.u = neurons.vector(mode = "uniform") + u        		# Tensor of recovery variables
		neurons.spike_traces = neurons.vector(mode = "zeros")
		neurons.spikes = neurons.vector(mode = "zeros")
		neurons.spike = neurons.vector(mode = "zeros") > 0

		# External input
		# neurons.voltage_i = (20 * neurons.vector(mode = 'uniform')) if voltage_i == 0.0 else (neurons.vector(mode = "zeros") + voltage_i) """ this doesnt work since i cannot set inputs of Initialize functions when i am sending attributes to the behaviour dinctionary when i am making an instance of this class. only default parameters are used. """
		# neurons.voltage_i = neurons.vector(mode = "zeros") + voltage_i
		neurons.voltage_i = 20 * neurons.vector(mode = 'uniform')

		# Model coefficients
		self.a = a                  # Time scale of the recovery variable
		self.b = b                  # Sensitivity of the recovery variable to the subthreshold fluctuations
		self.d = d                  # After-spike reset of the recovery variable
		self.c = c                  # After-spike reset value of membrane potential
		neurons.dt = dt				# dt = 0.5
		self.threshold = threshold  # Threshold for spike generation
		
		return 0

	def forward(self, neurons):
		voltage, u, a, b, c, d, voltage_max, dt = neurons.voltage, neurons.u, self.a, self.b, self.c, self.d, self.threshold, neurons.dt

		# Calculating the spikes
		spikes = voltage > voltage_max

		# Resetting membrane potential and recovery variable values of fired neurons
		if torch.sum(spikes) > 0:
			voltage[spikes] = c
			u[spikes] = u[spikes] + d

		# Calculating new values with time step dt = 0.5 for numerical stability
		voltage += dt * (0.04 * voltage**2 + 5.0 * voltage + 140 - u + neurons.voltage_i)
		u += dt * (a * (b * voltage - u))

		neurons.voltage = voltage
		neurons.spike = spikes.byte()
		neurons.u = u

		return 0

	# def swap_inputs(self):
	# 	self.voltage_i, self.voltage_i_next = self.voltage_i_next, self.voltage_i.zero_()

	# 	return 0

class Izhikevich_Neuron_Input(Izhikevich_Neuron):
	def initialize(self, neurons):
		return 0
