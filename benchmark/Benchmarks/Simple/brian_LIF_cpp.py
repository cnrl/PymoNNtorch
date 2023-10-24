from brian2 import *
import time
from globparams import *

set_device('cpp_standalone')

defaultclock.dt = 1*ms
prefs.core.default_float_dtype = float32
prefs.codegen.target = 'cython'

eqs_neurons = '''
dv/dt = (ge + rand() - v*OM_DECAY) / (1*ms) : 1
dge/dt = -ge / (1*ms) : 1
dspiked/dt = -spiked / (1*ms) : 1
'''

N = NeuronGroup(SIZE, eqs_neurons, threshold='v>VT', reset='v = VR', method='euler')

synaptic_model = '''
w : 1
'''

pre = '''
ge_post += w
spiked_pre = 1
'''

post = '''
w = clip(w + spiked_pre * STDP_SPEED, 0.0, 1.0) 
'''

S = Synapses(N, N, synaptic_model, on_pre=pre, on_post=post)

S.connect()
S.w = 'rand()/SIZE' #initialize
#S.w /= sum(S.w, axis=0) #normalize

if PLOT:
    M = SpikeMonitor(N)


run(DURATION*ms, report=REPORT_FUNC)


if PLOT:
    plot(M.t/ms, M.i, '.')
    show()
