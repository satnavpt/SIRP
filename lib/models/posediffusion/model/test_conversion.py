import torch
import numpy as np

def main():
    t = torch.tensor([[[ 1.0000, -0.0011, -0.0019],
         [ 0.0011,  1.0000,  0.0013],
         [ 0.0019, -0.0013,  1.0000]],

        [[-0.0133,  0.1998,  0.9797],
         [-0.1795,  0.9634, -0.1989],
         [-0.9837, -0.1785,  0.0231]]])

    print(t)
    print(t.permute(0, 1, 2))

main()

    # these come from the MFL dataset, and are define in the world -> camera format

    # the pose diffusion model operates on ndc coordinates, so we convert the init poses we use
