# Adapted from:
# https://github.com/mikecroucher/Jupyter-Matrix-Matrix

import timeit
import numpy
import cpuinfo


def bench_mmm(startn, maxn, step, loops, precision="double"):
    count = 0

    #Preallocate results lists
    m = int(1+(maxn-startn)/step)
    avg_gflops = m*[0]
    peak_gflops = m*[0]
    raw_times = [int(loops)*[0] for i in range(m)]
    all_gflops = [int(loops)*[0] for i in range(m)]
    mat_size = m*[0]

    for n in range(startn, maxn+step, step):
        setup_string = "import numpy as np; \
                n = 1000; \
                a = np.reshape(np.random.uniform(size=n*n), (n,n)); \
                b = np.reshape(np.random.uniform(size=n*n), (n,n)); \
                warmup = a.dot(b); \
                n = %d; \
                a = np.reshape(np.random.uniform(size=n*n), (n,n)); \
                b = np.reshape(np.random.uniform(size=n*n), (n,n))" % n
        setup_string += "; a = a.astype(np.%s); b = b.astype(np.%s)" % (
            precision, precision)
        time_list = timeit.repeat(
            "a.dot(b)", setup=setup_string, repeat=loops, number=1)
        raw_times[count] = time_list
        total_time = sum(time_list)
        avg_time = total_time / loops
        peak_time = min(time_list)
        num_ops = 2*n**3-n**2
        avg_gflops[count] = (num_ops/avg_time)/10**9
        peak_gflops[count] = (num_ops/peak_time)/10**9
        all_gflops[count] = [(num_ops/time)/10**9 for time in raw_times[count]]
        mat_size[count] = n
        print("N={} \t peak GFLOPS: {}".format(n, peak_gflops[count]))
        count = count+1

    import matplotlib.pyplot as plt
    plt.plot(mat_size, avg_gflops, '*-', label="Average over %d runs" % loops)
    plt.plot(mat_size, peak_gflops, '*-', label="Peak")
    plt.legend(bbox_to_anchor=(0.5, 0.2), loc=2, borderaxespad=0.)
    plt.xlabel('NxN Matrix Size, N')
    plt.ylabel('GFLOP/s, %s precision' % precision)
    plt.title('%s' % cpuinfo.get_cpu_info()["brand"])
    plt.savefig("mmflops-%s.pdf" % precision)

    return(max(peak_gflops), raw_times, all_gflops)
