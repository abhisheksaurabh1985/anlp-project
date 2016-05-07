import numpy as np

dtype = [('token',"S50"), ('postag', "S5"), ('real',"S2"), ('predicted',"S2")]
res = np.genfromtxt("test_out.csv", dtype=dtype, delimiter="	")
correct = sum(1 if (res['predicted'][i] == res['real'][i] and res['real'][i]!='O') else 0 for i in xrange(len(res['real'])))
total =  sum(1 if res['real'][i]!='O' else 0 for i in xrange(len(res['real'])))

print "captured non-O tags %f" % (correct/float(total))

correct = sum(1 if (res['predicted'][i] == res['real'][i]) else 0 for i in xrange(len(res['real'])))
total =  len(res['real'])

print "accuracy %f" % (correct/float(total))

class_res= np.zeros((5,5))
tags = ['O', 'BN', 'IN', 'BA', 'IA']
for k in xrange(5):
   for j in xrange(5):
      class_res[k,j] = sum(1 if (res['predicted'][i]==tags[j] and res['real'][i]==tags[k]) else 0 for i in xrange(len(res['real'])))

np.set_printoptions(precision=0, suppress=True)
print class_res
