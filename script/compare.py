import torch

file1 = "1.pt"
file2 = "2.pt"

def compare_pt_files(file1, file2):
    data1 = torch.load(file1, map_location="cpu")
    data2 = torch.load(file2, map_location="cpu")

    if data1.keys() != data2.keys():
        return "Keys mismatch"

    differences = {}
    for key in data1:
        if not torch.equal(data1[key], data2[key]):
            differences[key] = torch.abs(data1[key] - data2[key]).sum().item()

    return "No differences" if not differences else differences

result = compare_pt_files(file1, file2)
print(result)
