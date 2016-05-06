import numpy as np

dtype = [('token',"S50"), ('postag', "S5"), ('real',"S1"), ('predicted',"S1")]
res = np.genfromtxt("test_out.csv", dtype=dtype, delimiter="	")
correct = sum(1 if (res['predicted'][i]!='O' and res['real'][i]!='O') else 0 for i in xrange(len(res['real'])))
total =  sum(1 if res['real'][i]!='O' else 0 for i in xrange(len(res['real'])))

print "captured non-O tags %f" % (correct/float(total))

correct = sum(1 if (res['predicted'][i] == res['real'][i]) else 0 for i in xrange(len(res['real'])))
total =  len(res['real'])

print "accuracy %f" % (correct/float(total))

class_res= np.zeros((3,3))
class_res[0,0] = sum(1 if (res['predicted'][i]=='O' and res['real'][i]=='O') else 0 for i in xrange(len(res['real'])))
class_res[0,1] = sum(1 if (res['predicted'][i]=='B' and res['real'][i]=='O') else 0 for i in xrange(len(res['real'])))
class_res[0,2] = sum(1 if (res['predicted'][i]=='I' and res['real'][i]=='O') else 0 for i in xrange(len(res['real'])))

class_res[1,0] = sum(1 if (res['predicted'][i]=='O' and res['real'][i]=='B') else 0 for i in xrange(len(res['real'])))
class_res[1,1] = sum(1 if (res['predicted'][i]=='B' and res['real'][i]=='B') else 0 for i in xrange(len(res['real'])))
class_res[1,2] = sum(1 if (res['predicted'][i]=='I' and res['real'][i]=='B') else 0 for i in xrange(len(res['real'])))

class_res[2,0] = sum(1 if (res['predicted'][i]=='O' and res['real'][i]=='I') else 0 for i in xrange(len(res['real'])))
class_res[2,1] = sum(1 if (res['predicted'][i]=='B' and res['real'][i]=='I') else 0 for i in xrange(len(res['real'])))
class_res[2,2] = sum(1 if (res['predicted'][i]=='I' and res['real'][i]=='I') else 0 for i in xrange(len(res['real'])))

np.set_printoptions(precision=0, suppress=True)
print class_res
