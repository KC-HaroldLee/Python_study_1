import torch
import time

# start1 = time.time()
USE_CUDA = torch.cuda.is_available()
# print ('1. : {}'.format(time.time() - start1))

# start2 = time.time()
print('쿠다야 준비 되었니? - \'{}\'!'.format(USE_CUDA))
# print ('2. : {}'.format(time.time() - start2))

# start3 = time.time()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print ('3. : {}'.format(time.time() - start3))

start4 = time.time()
print('설마 cpu는 아니지? - \'{}\'!'.format(DEVICE))
print ('4. : {}'.format(time.time() - start4))