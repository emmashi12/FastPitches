import glob, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

in_filepath = '/Users/emmashi/Desktop/generated_pitch'
os.chdir(in_filepath)

for file in glob.glob("*.pt"):
    pitch_tensor = torch.load(file)
    pitch_tensor.squeeze()
    print(f'pitch_tensor shape: {pitch_tensor.shape}')
    
    image_name = 'f0_contour.png'
    plt.imshow(pitch_tensor)
    plt.show()
    plt.savefig(image_name)
