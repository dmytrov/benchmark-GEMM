import numpy
import mmflops

print('This was obtained using the following Numpy configuration:')
numpy.__config__.show()
sizes = list(range(100, 2500, 100)) 
sizes += [4, 8, 16, 32, 64, 128, 256, 512, 1024]
sizes.sort()
peak_flops = mmflops.bench_mmm(sizes, 3, precision="single")
