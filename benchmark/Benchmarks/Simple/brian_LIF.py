from brian2 import *
import time
from globparams import *

defaultclock.dt = 1*ms
prefs.core.default_float_dtype = float32

eqs_neurons = '''
dv/dt = (ge + rand() - v * OM_DECAY) / (1*ms) : 1
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


start = time.time()
run(DURATION*ms, report='text')
print("simulation time: ", time.time()-start)

if PLOT:
    plot(M.t/ms, M.i, '.')
    show()



#@network_operation(when='start', dt=10*ms)#, dt=10*ms
#def syn_norm():
#    print('test')
#    S2.w /= sum(S2.w, axis=0)

