# -*- coding: utf-8 -*-

import os
import pickle, gzip, random
'''        
        def train(dx, dy, valx,valy, n_epochs=1000):
            for epoch in range(n_epochs):
                avg_cost = train_model(dx,dy)
                precision = 100*validation(valx,valy)
                print("Epoch{}, cost={:.6f}, precision={:.2f}%".format(
                epoch, avg_cost, precision))
        def predict(dx):
            return self.predict_model(dx)
            
        def validation(dx,dy):
            # execute a prediction and check how many result are OK
            pred_y = self.predict(dx)
            nr_good = 0
            for i in range(len(dy)):
                if pred_y[i]==dy[i]:
                    nr_good+=1
            return 1.0*nr_good/len(dy)
'''

class FileDirectory(object):
    @staticmethod
    def find_file_path():
        # find the file contening the Data that we need
        root_dir = os.path.abspath('../..')
        data_dir = os.path.join(root_dir, 'Machine_Learning')
        return os.path.join(data_dir, 'files/')


        
class UnzipFiles(object):
    @staticmethod
    def unzip_files(fileName):
        path = FileDirectory.find_file_path()           
        f = gzip.open(path + fileName , 'rb')
        train_set, valid_set, test_set = pickle.load(f)        
        f.close()        
        return { 'trainSet': train_set,
                 'validSet': valid_set,
                 'testSet': test_set
               }

      
class Clean_data(object):     
    # ------------------------------------------------
    @staticmethod
    def get_cleaned_data(dataToClean, percetage = 0.8):
        outExtract = Clean_data.feature_extractor(dataToClean)
        n = int(len(outExtract['inputs'])*percetage)
        m = int((n - len(outExtract['inputs']))/2)
        return { 'trainSet': outExtract['inputs'][0:n], 
                 'trainVal': outExtract['targetLabel'][0:n], 
                 'trainValInIndex': outExtract['OutputIndex'][0:n],
                 'testSet': outExtract['inputs'][n:m], 
                 'testVal': outExtract['targetLabel'][n:m],
                 'testValIndex': outExtract['OutputIndex'][n:m],
                 'validSet': outExtract['inputs'][m:], 
                 'validVal': outExtract['targetLabel'][m:],
                 'validValIndex': outExtract['OutputIndex'][m:],
                 'realLabels': outExtract['labelName'],
                 'validInIndex': outExtract['OutputIndex'][0:n]                 
               }        
    @staticmethod
    def csv_file(files):
        path = FileDirectory.find_file_path() 
        with open(path  + files, 'r') as myfile :
            lines = myfile.readlines()
            
        return lines
    
    #-------------------------------------------------------------------------    
    # features extraction function
    @staticmethod
    def feature_extractor(feats, convert=True, list_in=[], list_out=[]):        
        L, Data_set = feats['lenght'], list() 
        files = feats['fileName'] + feats['fileExtension'] 
        
        path = FileDirectory.find_file_path() # get the file path
        with open(path  + files, 'r') as myfile :
            lines = myfile.readlines()
            Data_set = [l.strip().split(',') for l in lines if l.strip()]            
        random.shuffle(Data_set)        
        if (convert): # if we have categorical data 
            # This function convert the categorical data to numerical data
            dicategData = Clean_data.categorized_data(Data_set)
            #categorizedOutput = categData['categOutput'] 
            Data_set = dicategData['numOutput']            
            #categorizedOutput = self.CategorizedOutput
            L = len(dicategData['numOutput'][0])-len(dicategData['labelName'])
        ''' dataset[0] = ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
        les chiffres entre les '' sont consideres comme des chaines de 
        carateres donc il faut les Transformer en nombres '''               
        for line in xrange(len(Data_set)):
            for row in xrange(len(Data_set[line])):
                Data_set[line][row ] = float(Data_set[line][row])            
            list_in.append(Data_set[line][0:L])
            list_out.append(Data_set[line][L:])
                      
        return { 'inputs': list_in,
                 'targetLabel': list_out,
                 'OutputIndex': dicategData['OutputIndex'],
                 'labelName': dicategData['labelName'] 
               }
            
    # -------------------------------------------------------------------
    @staticmethod
    def categorized_data(categorized_Data, categorizedOutput = []): 
        for line, targetOutput  in enumerate(categorized_Data): 
            if targetOutput[-1] not in categorizedOutput:
                categorizedOutput.append(targetOutput[-1])
                # CategoryData = ['Iris-setosa', 'Iris-versicolor', ...]
        line, OutputIndex = 0, []
        while line < (len(categorized_Data)):                
            for cat in xrange(len(categorizedOutput)):
                if categorizedOutput[cat] in categorized_Data[line]:
                    # The category value is converted to numercal value in CategVect
                    CategVect = ['0']*len(categorizedOutput) 
                    # Get the position of the categorical output
                    Index = categorizedOutput.index(categorizedOutput[cat])
                    OutputIndex.append(Index) 
                    # Set the numerical output value to 1
                    CategVect[Index] = '1'                    
                    categorized_Data[line].pop() # Remove the categorical targeted value
                    # Remplace the categorical to new numerical output
                    categorized_Data[line].extend(CategVect)                    
                    break
            line+= 1            
        return {'numOutput': categorized_Data,
                'labelName': categorizedOutput,
                'OutputIndex': OutputIndex
               }
                   


        
        