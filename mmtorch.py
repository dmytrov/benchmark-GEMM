import torch
                                                                                                                                                                   
def op(a, b):                                                                                                                                             
    #return torch.logdet(a)
    #return torch.inverse(a)
    #return torch.eig(a)
    #return torch.cholesky(torch.mm(a, a.t()))
    return torch.mm(a, b)

torch.set_default_dtype(torch.float64)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
n = 1024*4
num_ops = 2*n**3-n**2
x = torch.randn(n, n)
y = torch.randn(n, n)
for i in range(2):
    start.record()
    z = op(x, y)
    end.record()
    torch.cuda.synchronize()
    print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CPU double")

torch.set_default_dtype(torch.float64)
device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
x = (torch.randn(n, n) + n*torch.eye(n)).to(device)
y = (torch.randn(n, n) + n*torch.eye(n)).to(device)
for i in range(5):
    start.record()
    z = op(x, y)
    end.record()
    torch.cuda.synchronize()
    print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CUDA double")

torch.set_default_dtype(torch.float32)
device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
x = (torch.randn(n, n) + n*torch.eye(n)).to(device)
y = (torch.randn(n, n) + n*torch.eye(n)).to(device)
for i in range(5):
    start.record()
    z = op(x, y)
    end.record()
    torch.cuda.synchronize()
    print(num_ops / start.elapsed_time(end) / 10**6, "GFLOPS CUDA single")
