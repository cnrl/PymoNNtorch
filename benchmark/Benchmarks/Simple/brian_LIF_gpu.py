from brian2 import *
import time
import platform
from globparams import *
import brian2cuda
#import brian2genn
#set_device('genn', use_GPU=True, debug=True)
set_device("cuda_standalone", clean=True)

defaultclock.dt = 1*ms
prefs.core.default_float_dtype = float32

if platform.node() == 'saeed-Swift-SF315-51G':
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 6.1
    prefs.devices.cuda_standalone.default_functions_integral_convertion = np.float32

vt = 6.1
vr = 0.0
input_strength = 1.0
stdp_speed = 0.001
decay = 1 - 0.9


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

