import h5py
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt

file = '/home/cl-ment-prigent/egodex/test/write/0.hdf5'
f = h5py.File(file, 'r')

print(f['transforms'].keys())