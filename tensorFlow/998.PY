# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import easyocr
import cv2 as cv
# print(cv.cuda.setDevice(device=0))
print(cv.cuda.getCudaEnabledDeviceCount())
cv.cuda.printCudaDeviceInfo(0)
print(cv.__version__)
# easyocr.Reader(lang_list=['ko'], gpu=True)