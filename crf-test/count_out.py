import numpy as np
import sys

if(len(sys.argv) <2 ):
   print "input file is not specified"

dtype = [('token',"S50"), ('postag', "S5"), ('real',"S2"), ('predicted',"S2")]
res = np.genfromtxt(sys.argv[1], dtype=dtype, delimiter="	")
correct = sum(1 if (res['predicted'][i] == res['real'][i] and res['real'][i]!='O') else 0 for i in xrange(len(res['real'])))
total =  sum(1 if res['real'][i]!='O' else 0 for i in xrange(len(res['real'])))

print "captured non-O tags %f" % (correct/float(total))

correct = sum(1 if (res['predicted'][i] == res['real'][i]) else 0 for i in xrange(len(res['real'])))
total =  len(res['real'])

print "accuracy %f" % (correct/float(total))

tags = ['O', 'B', 'I']
size = len(tags)
class_res= np.zeros((size,size))
for k in xrange(size):
   for j in xrange(size):
      class_res[k,j] = sum(1 if (res['predicted'][i]==tags[j] and res['real'][i]==tags[k]) else 0 for i in xrange(len(res['real'])))

np.set_printoptions(precision=0, suppress=True)
print class_res
