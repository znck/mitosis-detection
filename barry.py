import numpy as np
a = np.load('base_out.npy')
# b = np.load('model1_out.npy')
# c = np.load('model2_out.npy')
d = np.load('target.npy')
# a = a>0.5
# a[a>0.5]=1
# b[b>0.5]=1
# b[b<=0.5]=0
# c[c>0.5]=1
# c[c<=0.5]=0
print a
print d
import scipy.misc
scipy.misc.imsave('a.jpg',a)
scipy.misc.imsave('a_target.jpg', d)
# scipy.misc.imsave('b.jpg',b)
# scipy.misc.imsave('c.jpg',c)
