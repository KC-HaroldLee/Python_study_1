import torch
import time

USE_CUDA = torch.cuda.is_available()
print('쿠다야 준비 되었니? - \'{}\'!'.format(USE_CUDA))
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('설마 cpu는 아니지? - \'{}\'!'.format(DEVICE))