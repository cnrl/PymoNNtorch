from pymonntorch import *
from matplotlib import pyplot as plt
import torch

class Behavior1(Behavior):
    def initialize(self, ng):
        ng.spikes = ng.vector('rand') > 0.5

    def forward(self, blah):
        blah.spikes = blah.vector('rand') > 0.5

class Behavier2(Behavior):
    def initialize(self, ng):
        ng.trace = ng.vector()
        self.tau = self.parameter('tau', None)

    def forward(self, ng):
        ng.trace += (ng.spikes)*0.25 - ng.trace/self.tau

class VisModule(Behavior):
    def initialize(self, ng):
        # Window initilizer Async
        # name of varibale
        self.variable_name = self.parameter('var', None)
        t = getattr(ng, self.variable_name, None)
        pass

    def forward(self, ng):
        # get the Variable value
        t = getattr(ng, self.variable_name, None)
        # visualize
        # print(self.variable_name, ng.iteration, ": ")
        # print(t)

class SpikeOnTrace(Behavior):    
    def initialize(self, sg):
        sg.weights = sg.matrix("uniform") / sg.dst.size / 5

    def forward(self, sg):
        sg.dst.trace += torch.sum(sg.weights[sg.src.spikes], axis=0)

net = Network(dtype=torch.float32)

pop1 = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=100, width=100), behavior={
    1: Behavior1(), # behavior1(pop1)
    2: VisModule(var='spikes'),
    3: Behavier2(tau=10),

    9: VisModule(var='trace'),
    # 9: Recorder(variables=['trace'])
})

pop2 = NeuronGroup(net=net, size=NeuronDimension(depth=4, height=20, width=20), behavior={
    1: Behavior1(), # behavior1(pop1)
    2: VisModule(var='spikes'),
    3: Behavier2(tau=10),

    9: VisModule(var='trace'),
    # 9: Recorder(variables=['trace'])
})

pop3 = NeuronGroup(net=net, size=NeuronDimension(depth=1, height=10, width=10), behavior={
    1: Behavior1(), # behavior1(pop1)
    2: VisModule(var='spikes'),
    3: Behavier2(tau=10),

    9: VisModule(var='trace'),
    # 9: Recorder(variables=['trace'])
})

pop4 = NeuronGroup(net=net, size=NeuronDimension(depth=15, height=2, width=2), behavior={
    1: Behavior1(), # behavior1(pop1)
    2: VisModule(var='spikes'),
    3: Behavier2(tau=10),

    9: VisModule(var='trace'),
    # 9: Recorder(variables=['trace'])
})

# syn = SynapseGroup(net=net, src=pop1, dst=pop1, behavior={})

syn1_2 = SynapseGroup(net=net, src=pop1, dst=pop2, behavior={
    4: SpikeOnTrace(),
})

syn2_3 = SynapseGroup(net=net, src=pop2, dst=pop3, behavior={
    4: SpikeOnTrace(),
})

syn3_4 = SynapseGroup(net=net, src=pop3, dst=pop4, behavior={
    4: SpikeOnTrace(),
})


print("SSSSSSSS:",net.device)
net.initialize()

# net.simulate_iterations(100)


from visualizer.Visualize_OpenGL_IMGUI import GUI

GUI(net).initializeOpenGL()