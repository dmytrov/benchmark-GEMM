import numpy
import mmflops

print('This was obtained using the following Numpy configuration:')
numpy.__config__.show()
peak_flops = mmflops.bench_mmm(100, 2500, 100, 3, precision="single")
