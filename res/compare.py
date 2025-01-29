import torch

a = torch.load("./res/source2.pt")
b = torch.load("./res/source3.pt")
# b = torch.load("./res/image_syn.pt")

print(len(a))
print(len(a[0]))
print(len(a[0][0]))
# print(a==b)
# print(type(a))

# print(torch.allclose( a, b, atol=1e-4 ))

for i in range(len(a[0])):
    for j in range(len(a[0][0])):
        print(torch.allclose( a[0][i][j], b[0][i][j], atol=1e-4 ))
    

# import torch

# a = torch.load("res/student_params.pt")