# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:50:02 2016

@author: moussa
"""
import random
   
def solution1(arrayA):
    swap = 0
    for J in xrange(len(arrayA)):
        p = arrayA[J]
        for I in xrange(J+1, len(arrayA)):
            if p>arrayA[I]:
                swap+=1               
                p = arrayA[I]
                l = arrayA[J]
                arrayA[J] = p
                arrayA[I] = l
    if swap<=1:
        return True
    else:
        return False
    
def solution (S, T):
    # S = A2Le and 2bd?
    number = list('123456789')
    print number
    s, t = S.upper(), T.upper()
    text_s, text_S = list(s), list()
    text_t, text_T = list(t), list()
    for i in range(len(text_s)):
        if text_s[i] not in number: 
            text_S.append(text_s[i])
    
    for i in range(len(text_t)):
        if text_t[i] not in number: 
            text_T.append(text_t[i])
    L = min(len(text_S), len(text_T))
    
    itera = 0
    for k in range(L):
        if text_S[k] in text_T:
            itera+=1
    
    if itera>0:
        return True
    return False 
    
    '''
    s, t = S.upper(), T.upper()
    text_S = list(s)
    text_T = list(t)
    for i in range(len(text_S)):
        if text_S[i] in number:
            del text_S[i]
    
    for i in range(len(text_T)):
        if text_T[i] in number:
            del text_T[i]
            
    L = min(len(text_S), len(text_T))
    itera = 0
    for k in range(L):
        if text_S[k] in text_T:
            itera+=1
    
    if itera>0:
        return True
    return False            
    '''
                         
if __name__=="__main__": 
    N = 10
    A = [i for i in xrange(N)]
    random.shuffle(A)
    S = 'A2ple'
    T = '2df1'
    
    print solution(S, T) # return False
    
     
                # print 'Optimization Finished!'
                # print 
                # Test model
                
                # test_accuracy = accuracy.eval({inputs:Test_x, outputs: Test_y})
                # print "Epoch:", '%04d' % (epoch),", accuracy=", "{:.3f}".format(test_accuracy), ", cost=", "{:.3f}".format(np.mean(train_accuracy))
                    
                '''    
                cost = 0
                # TRAINING PHASE
                for batch_x, batch_y in zip(batches_x, batches_y):
                    avg_mini_cost = 0
                    for x, y in zip(batch_x, batch_y):                    
                        _, y_pred, c = sess.run([optimizer, output_layer, cost_mean], {inputs: np.array([x]), outputs: y})
                        avg_mini_cost+= c                        
                        #print y, '==', y_pred                       
                    cost+= avg_mini_cost/len(batch_x)                    
                avg_tr_cost.append(cost/len(batches_x))
                
                # EVALUATION PHASE
                accuracy = 0
                test_Cost = 0
                if Test_x!=None:                    
                    for X, Y in zip(Test_x, Test_y):
                        accu, test_cost = sess.run([Accuracy, cost_mean], {inputs: np.array([X]), outputs: Y})
                        accuracy+= accu
                        test_Cost+= test_cost 
                    avg_accuracy.append(accuracy/len(Test_x))
                    avg_test_cost.append(test_Cost/len(Test_x))
                    print '======', accuracy/len(Test_x) 
                '''

        
    

    
    