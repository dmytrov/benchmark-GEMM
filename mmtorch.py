import torch
torch.set_default_dtype(torch.float64)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n = 1000
x = torch.randn(n, n)
y = torch.randn(n, n)
start.record()
z = torch.mm(x, y)
end.record()
torch.cuda.synchronize()
num_ops = 2*n**3-n**2
print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CPU double")


import torch
torch.set_default_dtype(torch.float32)
device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n = 10000
x = torch.randn(n, n).to(device)
y = torch.randn(n, n).to(device)
for i in range(10):
    start.record()
    z = torch.mm(x, y)
    end.record()
    torch.cuda.synchronize()
    num_ops = 2*n**3-n**2
    print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CUDA single")


import torch
torch.set_default_dtype(torch.float64)
device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n = 1024*4
x = torch.randn(n, n).to(device)
y = torch.randn(n, n).to(device)
for i in range(10):
    start.record()
    z = torch.mm(x, y)
    end.record()
    torch.cuda.synchronize()
    num_ops = 2*n**3-n**2
    print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CUDA double")
