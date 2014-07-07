from pylab import *

n = arange(0,512)
x = fromfile("fake_fourier_in.bin",dtype="float32")
y = fromfile("rfft_out.bin",dtype="float32")

rfftx = rfft(x)
print len(rfftx)
plot(arange(0,513),rfftx**2,n,y)
show()
