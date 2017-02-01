# -*- coding: utf-8 -*-
# UNSUPERVISED ALGORITHM

import tensorflow as tf
import numpy as np
# IMPORT CLASSES
import getData as getd

class LamdaClass(object):
    @staticmethod
    def newClass(X, classIndex):
        rho_Ci, Ci_NIC = np.zeros((len(X),1)), 0.5*np.ones((len(X),1))
        for k in xrange(len(X)):
            rho_Ci[k,0] = Ci_NIC[k,0] + (X[k]-Ci_NIC[k,0])
        rho = {'C%s'%classIndex: rho_Ci}
        classesData = {'C%s'%classIndex:[X]}         
        return rho, classesData
        
    @staticmethod    
    def lamda(Data):
        init = True        
        for X in Data:
            L = len(X)
            if init:
                rho, classesData = LamdaClass.newClass(X,0)
                init = False
            else:
                Nc = len(classesData)
                # DEGREE OF MARGINAL ADEQUACY (DMA), THIS IS A PROBABILITY
                DAG = []
                for cls in xrange(Nc):
                    prod = 1
                    for i in xrange(L):
                        DMA_i_cls = (rho['C%s'%(cls)][i]**(X[i]))*((1-rho['C%s'%(cls)][i])**(1-X[i]))
                        prod = prod*DMA_i_cls
                    DAG.append(prod)                                                                   
                # DEGREE OF ADEQUACY GLOBAL DAG, THIS IS A PROBABILITY
                Ci_NIC = 0.5*np.ones((L, 1))
                prod_NIC = 1
                for j in xrange(L):
                    # THE NIC CLASS
                    prod_NIC = prod_NIC*Ci_NIC[j,0]                        
                # DEGREE OF MARGINAL ADEQUACY 
                DAG.append(prod_NIC)
            
                # AFFECTATION PHASE
                class_index = np.argmax(DAG)
                if (class_index<(len(DAG)-1)):
                    print 'Allo'
                    classesData['C%s'%class_index].append(X) 
                    i = 0
                    while i<len(X):
                        # UPDATE THE rho ASSOCIATED TO THE CLASS WHOSE THE INDEX IS class_index
                        N = len(classesData['C%s'%class_index])  
                        #rho_i = rho['C%s'%class_index][i,0]
                        rho['C%s'%class_index][i,0] = rho['C%s'%class_index][i,0]+(X[i]-rho['C%s'%class_index][i,0])/N
                        i += 1                    
                else: # A NEW CLASS IS DETECTED
                    newrho, newClass = LamdaClass.newClass(X,class_index)
                    classesData.update(newClass)
                    rho.update(newrho)
                #print 'Len C', len(classesData['C0'])    
        #print len(classesData)        
            
if __name__=="__main__":
    Data = getd.CvsFileMnist. manupulations(10)

    LamdaClass.lamda(Data['train_x'])

    
    

