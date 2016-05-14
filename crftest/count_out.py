import numpy as np
import sys



def count_out(filename, verbose = False):
   dtype = [('token',"S50"), ('postag', "S5"), ('triggers', 'S7'),('isPunct', 'S5'),
            ('chunk', 'S5'),('segment', 'S5'),('concept', 'S5'),
            ('real',"S2"), ('predicted',"S2")]
   res = np.genfromtxt(filename, dtype=dtype, delimiter="	")
   correct = sum(1 if (res['predicted'][i] == res['real'][i] and res['real'][i]!='O') else 0 for i in xrange(len(res['real'])))
   total =  sum(1 if res['real'][i]!='O' else 0 for i in xrange(len(res['real'])))

   nonOTagsAccuracy = (correct/float(total))
   if verbose: print "captured non-O tags %f" % nonOTagsAccuracy

   correct = sum(1 if (res['predicted'][i] == res['real'][i]) else 0 for i in xrange(len(res['real'])))
   total =  len(res['real'])

   accuracy = (correct/float(total))
   if verbose: print "accuracy %f" % accuracy

   tags = ['-', 'N', 'A']
   size = len(tags)
   class_res= np.zeros((size,size))
   for k in xrange(size):
      for j in xrange(size):
         class_res[k,j] = sum(1 if (res['predicted'][i]==tags[j] and res['real'][i]==tags[k]) else 0 for i in xrange(len(res['real'])))

   np.set_printoptions(precision=0, suppress=True)
   if verbose: print class_res

   return (nonOTagsAccuracy, accuracy, class_res)


if __name__ == "__main__":

   if(len(sys.argv) <2 ):
      print "input file is not specified"
   count_out(sys.argv[1], True)
