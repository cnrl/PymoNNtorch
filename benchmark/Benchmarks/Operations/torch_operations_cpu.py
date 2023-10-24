import torch
import time

#heating up the CPU
#for i in range(1000):
#    temp = np.random.rand(10000, 5000)*np.random.rand(10000, 5000)+np.random.rand(10000, 5000)

d = 'cpu'
t = torch.float64

#for i in range(60):
for i in range(1):
    measurements = []

    steps = 1000

    ###########################################
    print('Initialization...')
    ###########################################

    src = torch.rand(5000, device=d) < 0.01  # d = 'cpu' or 'gpu'
    dst = torch.rand(10000, device=d) < 0.01
    W1 = torch.rand(10000, 5000, device=d, dtype=t)
    W2 = torch.rand(5000, 10000, device=d, dtype=t)

    ###########################################
    print('\nSynapse Operation...')
    ###########################################

    start = time.time()
    for i in range(steps):
        torch.tensordot(W1, src.to(t), dims=([1],[0]))
    t1 = (time.time()-start)/steps*1000
    print('  W1.dot(s):', t1, 'ms')
    measurements.append(t1)


    start = time.time()
    for i in range(steps):
        torch.sum(W1[:, src], dim=1)
    t2 = (time.time()-start)/steps*1000
    print('  np.sum(W1[:, s], axis=1):', t2, 'ms', t1/t2, 'x ratio')
    measurements.append(t2)


    # same but with W2 (SxD) instead of W1 (DxS):


    start = time.time()
    for i in range(steps):
        torch.tensordot(W2.T, src.to(t), dims=([1], [0]))
        #W2.T.dot(src)
    t3 = (time.time()-start)/steps*1000
    print('  W2.T.dot(s):', t3, 'ms', t1/t3, 'x ratio')
    measurements.append(t3)


    start = time.time()
    for i in range(steps):
        torch.sum(W2[src], axis=0)
    t4 = (time.time()-start)/steps*1000
    print('  np.sum(W2[s], axis=0):', t4, 'ms', t1/t4, 'x ratio')
    measurements.append(t4)


    ###########################################
    print('\nSTDP...')
    ###########################################

    start = time.time()
    for i in range(steps):
        W1 += dst[:, None] * src[None, :]
    t1 = (time.time()-start)/steps*1000
    print('  W1 += d[:, None] * s[None, :]:', t1, 'ms')
    measurements.append(t1)


    #W1[d, s] += 1 # ERROR!


    start = time.time()
    for i in range(steps):
        W1[dst[:, None] * src[None, :]] += 1
    t2 = (time.time()-start)/steps*1000
    print('  W1[d[:, None] * s[None, :]] += 1:', t2, 'ms', t1/t2, 'x ratio')
    measurements.append(t2)


    start = time.time()
    for i in range(steps):
        W1[(torch.where(dst)[0].view(-1, 1), torch.where(src)[0].view(1, -1))] += 1
        #W1[np.ix_(dst, src)] += 1
    t3 = (time.time()-start)/steps*1000
    print('  W1[np.ix_(d, s)] += 1:', t3, 'ms', t1/t3, 'x ratio')
    measurements.append(t3)

    #same but with W2 (SxD) instead of W1 (DxS):

    start = time.time()
    for i in range(steps):
        W2 += src[:, None] * dst[None, :]
    t1 = (time.time()-start)/steps*1000
    print('  W2 += s[:, None] * d[None, :]:', t1, 'ms')
    measurements.append(t1)


    #W1[d, s] += 1 # ERROR!


    start = time.time()
    for i in range(steps):
        W2[src[:, None] * dst[None, :]] += 1
    t2 = (time.time()-start)/steps*1000
    print('  W2[s[:, None] * d[None, :]] += 1:', t2, 'ms', t1/t2, 'x ratio')
    measurements.append(t2)


    start = time.time()
    for i in range(steps):
        W2[(torch.where(src)[0].view(-1, 1), torch.where(dst)[0].view(1, -1))] += 1
        #W2[np.ix_(src, dst)] += 1
    t3 = (time.time()-start)/steps*1000
    print('  W2[np.ix_(s, d)] += 1:', t3, 'ms', t1/t3, 'x ratio')
    measurements.append(t3)



    ###########################################
    print('\nReset operation...')
    ###########################################

    steps = 100000
    voltage = torch.rand(5000, device=d, dtype=t)

    start = time.time()
    for i in range(steps):
        voltage = voltage * 0.0
    t1 = (time.time()-start)/steps*1000
    print('  voltage = voltage * 0.0:', t1, 'ms')
    measurements.append(t1)


    start = time.time()
    for i in range(steps):
        voltage = torch.zeros(5000, device=d, dtype=t)
    t2 = (time.time()-start)/steps*1000
    print('  voltage = torch.zeros(5000, dtype=dtype):', t2, 'ms', t1/t2, 'x ratio')
    measurements.append(t2)


    start = time.time()
    for i in range(steps):
        voltage.fill_(0)
    t3 = (time.time()-start)/steps*1000
    print('  voltage.fill(0):', t3, 'ms', t1/t3, 'x ratio')
    measurements.append(t3)


    ###########################################
    print('\nDatatypes...')
    ###########################################
    steps = 1000

    W2 = W2.to(torch.float64)
    start = time.time()
    for i in range(steps):
        torch.sum(W2[src], axis=0)
    t1 = (time.time()-start)/steps*1000
    print('  float64:', t1, 'ms')
    measurements.append(t1)


    W2 = W2.to(torch.float32)
    start = time.time()
    for i in range(steps):
        torch.sum(W2[src], axis=0)
    t2 = (time.time()-start)/steps*1000
    print('  float32:', t2, 'ms', t1/t2, 'x ratio')
    measurements.append(t2)


    W2 = W2.to(torch.float16)
    start = time.time()
    for i in range(steps):
        torch.sum(W2[src], axis=0)
    t3 = (time.time()-start)/steps*1000
    print('  float16:', t3, 'ms', t1/t3, 'x ratio')
    measurements.append(t3)