import torch

file1 = "./script/a.pt"
file2 = "./script/b.pt"

def compare_pt_files(file1, file2):
    data1 = torch.load(file1, map_location="cpu")
    data2 = torch.load(file2, map_location="cpu")

    # print(data1)
    print(torch.equal(data1,data2))
    # if data1.keys() != data2.keys():
    #     return "Keys mismatch"

    # differences = {}
    # # for key in data1:
    # if not torch.equal(data1, data2):
    #     differences[key] = torch.abs(data1[key] - data2[key]).sum().item()

    # return "No differences" if not differences else differences

result = compare_pt_files(file1, file2)
# print(result)
