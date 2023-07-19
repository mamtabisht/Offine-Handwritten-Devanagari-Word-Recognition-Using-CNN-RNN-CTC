
import cv2
import numpy as np
import keras.backend as K
char_list = 'अआइईउऊएऐओऔऋकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह०१२३४५६७८९ािीुूेैोौंःृँ़् '
 #   len(char_list)=70
 
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst
        

#Testing
path = 'D:/IIIT-HW-Dev/HindiSeg/HindiOp2/test'
#lists for test dataset
test_img = []
test_txt = []
test_input_length = []
test_label_length = []
test_orig_txt = []

f=open(path+'/'+'test.txt', encoding="utf8")
for line in f:
    print(line)    
    txt=line.strip().split(' ')[1]
    test_orig_txt.append(txt) ; 
    test_label_length.append(len(str(txt))); 
    test_input_length.append(31); 
    test_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
            
    img_path_=line.split(' ')[0].split('/')[2:5]
    img_path=(path+'/'+img_path_[0]+'/'+img_path_[1]+'/'+img_path_[2])       
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)      
     #convert each image of shape (32, 128, 1)
    (w, h) = img.shape
    if w>32 or h>128:
        #continue
        img = cv2.resize(img,(128,32))
    if w < 32:
        add_zeros = np.ones((32-w, h))*255
        img = np.concatenate((img, add_zeros))
    if h < 128:
        add_zeros = np.ones((32, 128-h))*255
        img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img , axis = 2)
            # Normalize each image
    img = img/255.
        
    test_img.append(img)

test_img = np.array(test_img)
##########################################3    

# load the saved best model weights
from keras.models import load_model
act_model_Dev = load_model('act_model_Dev-.hdf5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
act_model_Dev.load_weights('Best-weights-131-1.9525.hdf5')
act_model_Dev.summary()

#Evaluate handwritten text recognition results using Best Path/Greedy
#without lexicon correction
prediction = act_model_Dev.predict(test_img[:])
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

listt=[]
for i in range(len(out)):
    #print(i)
    lst=[]
    for x in out[i]:
        #print(x)
        if x!=-1:
            lst.append(x)
    listt.append(lst) 

#Best Path/Greedy
#without lexicon correction    
numWordOK = 0 
numWordTotal = 0 
numCharTotal = 0  
numCharErr = 0  
numCharOK = 0
for i in range(len(listt)):
    if np.array_equal(listt[i],test_txt[i])== True:
        numWordOK+=1
    numWordTotal+=1  
    minLength=min(len(test_txt[i]),len(listt[i]))
    for j in range(minLength):
        if np.array_equal(test_txt[i][j], listt[i][j])== True:
            numCharOK+=1
    numCharTotal+=len(test_txt[i])
    
numCharErr= numCharTotal-numCharOK
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
WER=(numWordTotal-numWordOK)/numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%. WER: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0, WER*100.0))
    
##############################################################
#Take lexicon  
lexicon_txt=[]
test_lexicon_txt=[]

lexicon_path='D:/IIIT-HW-Dev/lexicon.txt' 
f_lexicon=open(lexicon_path, encoding="utf8")
for line in f_lexicon:
    #print(line)    
    txt=line.strip() 
    lexicon_txt.append(txt) ; 
    test_lexicon_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
    
###########################################################################    
#correct Best Path/Greedy results with lexicon correction
#append it in List as listtt   
start_time = datetime.datetime.now()
listtt=[]
import editdistance
numWordOK = 0 
numWordTotal = 0 
numCharTotal = 0  
numCharErr = 0  
numCharOK = 0
num_Lexicon_corrected=0
numCharTotal = 0
numCharErr = 0 

#for i in range(len(listt)):
#take counter 1-12866 to consider all test samples
for i in range(1,12865):
    print(i)
    numCharTotal+=len(test_txt[i])
    if np.array_equal(listt[i],test_txt[i])== True:
        numWordOK+=1
        listtt.append(listt[i])
        
    else:
        distance=[]
        
        for k in range(len(test_lexicon_txt)):
            dist =editdistance.eval(np.array(listt[i]), np.array(test_lexicon_txt[k]))
            #distance_array = np.array(distance)
            distance.append(dist)
            temp =np.min(distance) 
            #find index of minimun dist
            res = [] 
            for idx in range(0, len(distance)): 
                if temp == distance[idx]: 
                    res.append(idx)
                   # print("The Positions of minimum element : " + str(res)) 
                    #The Positions of minimum element : [6101, 6267, 8583]        
        #to do: if res[] has more than 1 element code for lexicon correction
            
        listtt.append(test_lexicon_txt[res[0]])
        num_Lexicon_corrected+=1   
    
    numWordTotal+=1  
total_time = datetime.datetime.now() - start_time    
##############################
  
#results evaluation after lexicon correction in Best Path/Greedy algo
numWordOK = 0 
numWordTotal = 0 
numCharTotal = 0  
numCharErr = 0  
numCharOK = 0
#for i in range(len(listtt)):
m=0
for i in range(1,12865):
    #print(i)
    
    if np.array_equal(listtt[m],test_txt[i])== True:
        numWordOK+=1
    numWordTotal+=1  
    minLength=min(len(test_txt[i]),len(listtt[m]))
    for j in range(minLength):
        if np.array_equal(test_txt[i][j], listtt[m][j])== True:
            numCharOK+=1
    numCharTotal+=len(test_txt[i])
    m+=1
numCharErr= numCharTotal-numCharOK
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
WER=(numWordTotal-numWordOK)/numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%. WER: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0, WER*100.0))

###################################  
  
# for single sample beam search algo
print("original_text =  ", test_orig_txt[10:11]) #बना
prediction1 = act_model_Dev.predict(test_img[0:1])
top_paths = 3
results1 = []
for i in range(top_paths):
    lables1 = K.get_value(K.ctc_decode(prediction1, input_length=np.ones(prediction1.shape[0])*prediction1.shape[1],
                       greedy=False, beam_width=top_paths, top_paths=top_paths)[0][i])[0]
    results1.append(lables1)
for p in results1: 
   for y in p:
      if int(y) != -1:
          print(char_list[int(y)], end = '')       
   print('\n')         
###############################
#beam search algo.. using numpy #append beam search results in list as Beam_S_List
#prediction = act_model_Dev.predict(test_img[:100]) #print("original_text =  ", test_orig_txt[:5])

import datetime   
start_time = datetime.datetime.now()
Beam_S_List=[]
for m in range(10000,12865):     
    print(m)
    
    prediction = act_model_Dev.predict(test_img[m:m+1])
    top_paths = 3
    results = []
    for i in range(top_paths):
       
        lables = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                       greedy=False, beam_width=top_paths, top_paths=top_paths)[0][i])[0]
        results.append(lables)
    Beam_S_List.append(results)
total_time = datetime.datetime.now() - start_time 
     
#############################
    
#calculate recognition results using Beam Search algo and without lexicon correction    
import numpy as np
#import editdistance 
numCharOK=0   
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0  
tst=[]
tst=test_txt[10000:12865] 
      
print('Ground truth -> Recognized')	
for i in range(len(Beam_S_List)):
    #print(i)
    numCharOK=0
    numCharErrErr=0
   
    if np.array_equal(tst[i] ,Beam_S_List[i][0])== True:
        numWordOK += 1
     
    elif np.array_equal(tst[i] ,Beam_S_List[i][1])== True:
        numWordOK += 1  
        
    elif np.array_equal(tst[i] ,Beam_S_List[i][2])== True:
        numWordOK += 1
            #listtt_beam.append(Beam_S_List[i][k].tolist())
    else:
        print(i)   #print incorrect words index
        numCharOK0=0    
        minLength0=min(len(tst[i]),len(Beam_S_List[i][0]))
        for j in range(minLength0):
            #print(j)
            if np.array_equal(tst[i][j], (Beam_S_List[i][0])[j])== True:
                numCharOK0+=1
                
        numCharOK1=0
        minLength1=min(len(tst[i]),len(Beam_S_List[i][1]))
        for j in range(minLength1):
            if np.array_equal(tst[i][j], (Beam_S_List[i][1])[j])== True:
                numCharOK1+=1 
                
        numCharOK2=0
        minLength2=min(len(tst[i]),len(Beam_S_List[i][2]))
        for j in range(minLength2):
            if np.array_equal(tst[i][j], (Beam_S_List[i][2])[j])== True:
                numCharOK2+=1       
                
        #numCharOK+=max(numCharOK0, numCharOK1, numCharOK2)
        numCharOK=max(numCharOK0, numCharOK1, numCharOK2)
        numCharErrErr= len(tst[i])-numCharOK
    numCharErr+= numCharErrErr
    numCharTotal+=len(tst[i])
        
    numWordTotal+=1  
#calculate results before lexicon correction 
#numCharErr= numCharTotal-numCharOK
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
WER=(numWordTotal-numWordOK)/numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%. WER: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0, WER*100.0))
        
############################################

#correct Beam-Search algo results with lexicon
import datetime 
start_time = datetime.datetime.now()
import editdistance    
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0  
num_Lexicon_corrected=0     
listtt_beam=[]
tst=[]
tst1=test_txt[10000:12865] 
tst=test_lexicon_txt[:] 
#tst=valid_txt[:10] 
      
print('Ground truth -> Recognized')	
for i in range(len(Beam_S_List[:])):
    print(i)
    for k in range(3):
        if tst[i] == (Beam_S_List[i][k].tolist()):
            numWordOK += 1
            listtt_beam.append(Beam_S_List[i][k].tolist())
        else:
            # correct predicted listt[] with lexicon
            distance=[]  
            #print(i) 
            for j in range(len(tst)):
                #print(j)
                dist = (editdistance.eval(Beam_S_List[i][k].tolist(), tst[j]))
                distance.append(dist)
                #distance_array = np.array(distance)
                temp = min(distance) 
                #find index of minimun dist
                res = [] 
                for idx in range(0, len(distance)): 
                    if temp == distance[idx]: 
                        res.append(idx)
                   # print("The Positions of minimum element : " + str(res)) 
                    #The Positions of minimum element : [6101, 6267, 8583]        
                    #to do: if res[] has more than 1 element code for lexicon correction
            listtt_beam.append(tst[res[0]])
          
            num_Lexicon_corrected+=1            
                         
    numWordTotal += 1   
    numCharErr+=dist
    #numCharTotal += len(tst[i])
    numCharTotal += len(tst1[i])
total_time = datetime.datetime.now() - start_time 

# select unique items in listtt_beam     
    
output = []
zz=0
#for i in range(2000):
#for i in range(len(tst[:100])):
for i in range(len(Beam_S_List)):
    listtt_beam_2=[]
    output1 = []
    listtt_beam_2= listtt_beam[zz:zz+3]
    for x in listtt_beam_2:
        #print(x)
        if x not in output1:
            output1.append(x) 
    output.append(output1)     
    zz+=3   

#Evaluate Beam search results after lexicon correction. 
    
# Output list contain unique lexicon corrected Beam-search algo results of trained model on test data.
#if unique Output list contain single element in each cell, then it is directly compare with actual label
#but if it contain 2 or 3 list items in single cell, then take the one which give minimun error with actual label   
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=test_txt[10000:12865]        
print('Ground truth -> Recognized')	
for i in range(len(output)):  # listtt is lexicon corrected
#for i in range(10):
    numCharTotal += len(tst[i])

    if len(output[i])==1:
        #print("length is one")
        #print(i)
        if ([tst[i]] == output[i]):
            numWordOK += 1
            #print("TRUE")
        else:
             numWordOK += 0   
        
    elif len(output[i])==2:
        #print("length is two")
        #print(i)
        if (tst[i] == output[i][0]) or (tst[i] == output[i][1]):
            numWordOK += 1 
            #print("TRUE")
        else:
            numWordOK += 0         
            
    else : # if len(output[i])==3:
        #print("length is three")
        #print(i)
        if (tst[i] == output[i][0]) or (tst[i] == output[i][1]) or (tst[i] == output[i][2]):
            numWordOK += 1 
            #print("TRUE")
        else:
            numWordOK += 0
        
    numWordTotal += 1 
    distance=[]  
    for y in range(len(output[i])):
        dist = editdistance.eval(output[i][y], tst[i])
        distance.append(dist)
        #distance_array = np.array(distance)
        temp = min(distance)    
        
        numCharErr+=temp
        #numCharTotal += len(tst[i])
    
        if dist==0:
            print('[OK]') 
            '''
        else: 
        
            for p in tst[i]:  
                print(char_list[int(p)], end = '') 
            print('\n')
            for z in range(len(output[i])):    
                for p in output[i][z]:  
                    print(char_list[int(p)], end = '') 
                print('\n')
'''

#calculate results with Beam search and lexicon correction together

charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
WER=(numWordTotal-numWordOK)/numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%. WER: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0, WER*100.0))

#####################################

# code to calculate total number of charcters
numCharTotal11=0
for m in range(2000):     
    #print(m)
    numCharTotal11 += len(tst[m])
########################################

# prediction on other dataset, Indic Word IIT roorkee
#Testing
path = 'D:/IndicWord/a02'
#lists for test dataset
test_img = []
test_txt = []
test_input_length = []
test_label_length = []
test_orig_txt = []

f=open(path+'/'+'val.txt', encoding="utf-8-sig")
for line in f:
    print(line)    
    txt=line.strip().split(' ')[8]
    test_orig_txt.append(txt) ; 
    test_label_length.append(len(str(txt))); 
    test_input_length.append(31); 
    test_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
            
    img_path_=line.split(' ')[0]
    img_path_=(img_path_+'.jpg'); print(img_path_)
    img_path=(path+'/'+img_path_);print(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)      
     #convert each image of shape (32, 128, 1)
    img = cv2.resize(img,(128,32))
    img = np.expand_dims(img , axis = 2)
    # Normalize each image
    img = img/255.
    test_img.append(img)#; print(valid_img)
test_img = np.array(test_img) 
################################################

