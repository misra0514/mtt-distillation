import torch
from StreamBind import bind

a = torch.rand(2,3)

# with torch.cuda.stream:
# curr_stm = torch.cuda.current_stream()
# print(curr_stm.cuda_stream)

    
bind(10,0,a )
