import torch
x = torch.tensor(3.5)
y = x*x + 2
print(x, y)

# grad
x = torch.tensor(3.5, requires_grad=True)
print(x)
y = (x-3) * (x-1) * (x-2)
print(y)
y.backward()
print(x.grad)

# chain_rule
x = torch.tensor(3.5, requires_grad=True)
y = x*x
z = 2*y + 3

z.backward()
print(x.grad)

# chain_rule_2
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()
print(a.grad)
print(b.grad)
