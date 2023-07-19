
#on IIIT hyderabad Devanagari word Dataset


import os
import fnmatch
import cv2
import numpy as np
import string
import time
import datetime

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Dropout
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt
from keras import callbacks

import tensorflow as tf

#ignore warnings in the output
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras.regularizers import l2

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
        
########################################################
# Train
 # lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

path = 'D:/IIIT-HW-Dev/HindiSeg/HindiOp2/train'
f=open(path+'/'+'train.txt', encoding="utf8")
for line in f:
    print(line)    
    txt=line.strip().split(' ')[1]
    orig_txt.append(txt) ; 
    train_label_length.append(len(str(txt))); 
    train_input_length.append(31); 
    training_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
            
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
        
    training_img.append(img)          
            
##################################################
#Validation
 #lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

path = 'D:/IIIT-HW-Dev/HindiSeg/HindiOp2/val'

f=open(path+'/'+'val.txt', encoding="utf8")
for line in f:
    print(line)    
    txt=line.strip().split(' ')[1]
    valid_orig_txt.append(txt) ; 
    valid_label_length.append(len(str(txt))); 
    valid_input_length.append(31); 
    valid_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
            
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
        
    valid_img.append(img)
    
#######################################
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
                  
    img = img/255.
        
    test_img.append(img)  
###############################################################################

max_label_len= max(train_label_length)
print(max_label_len )  #23

max_label_len= max( test_label_length)
print(max_label_len )            #22

max_label_len= max( valid_label_length)
print(max_label_len )            #23

# pad each output label to maximum text length : choose max_label_len=23
train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))    
test_padded_txt = pad_sequences(test_txt, maxlen=max_label_len, padding='post', value = len(char_list))   
 
###############################################################################
 

# input with height=32 and width=128 
inputs = Input(shape=(32,128,1))
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(32, (3,3), activation = 'relu',kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
conv_2 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
pool_2=(Dropout(0.2))(pool_2)

conv_3 = Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(pool_2)
conv_4 = Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
pool_4=(Dropout(0.3))(pool_4)
 
conv_5 = Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
pool_6=(Dropout(0.4))(pool_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu',  kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(pool_6)
pool_7 = MaxPool2D(pool_size=(2, 1))(conv_7 )
pool_7=(Dropout(0.4))(pool_7)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(pool_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model_Dev = Model(inputs, outputs)    
act_model_Dev.summary() 

act_model_Dev.save('act_model_Dev-.hdf5')

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
 
 
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
     
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.save('model_Dev.hdf5')
model.summary()


model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'Adam')
 

filepath="Best-weights-{epoch:03d}-{loss:.4f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
filename='train.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
callbacks_list = [checkpoint, csv_log]


training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

test_img = np.array(test_img)
test_input_length = np.array(test_input_length)
test_label_length = np.array(test_label_length)


batch_size = 100
epochs = 150


start_time = datetime.datetime.now()
history = model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs, validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)
total_time = datetime.datetime.now() - start_time
print(total_time)


# load the saved best model weights
from keras.models import load_model
act_model_Dev = load_model('act_model_Dev-.hdf5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
act_model_Dev.load_weights('Best-weights-131-1.9525.hdf5')
act_model_Dev.summary()


# predict outputs on validation images
prediction = act_model_Dev.predict(valid_img[:20])
# predict outputs on test images
prediction = act_model_Dev.predict(test_img[:])



# use CTC decoder

out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
# see the results  

i = 0
for x in out:
    print("original_text =  ", valid_orig_txt[i])#:test_orig_txt[i])
    print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')       
    print('\n')
    i+=1
      
####################
#code to make list, subset of another list after droping some elements. 
#in loop 
listt=[]
for i in range(len(out)):
    #print(i)
    lst=[]
    for x in out[i]:
        #print(x)
        if x!=-1:
            lst.append(x)
    listt.append(lst)      
#print(listt)            
    
#####################
# without lexicon correction
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=test_txt[:]
#tst=valid_txt[:10]        
print('Ground truth -> Recognized')	
for i in range(len(listt)):
    #print(i)
    if len(listt[i])==len(tst[i]):        
        if tst[i] == listt[i]:             
            numWordOK += 1
        else:
            numWordOK += 0
    else:
        numWordOK += 0 
        
    numWordTotal += 1   
    dist = editdistance.eval(listt[i], tst[i])
    numCharErr+=dist
    numCharTotal += len(tst[i])
    
    if dist==0:
        print('[OK]') 
    else: 
        for p in tst[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')
        for p in listt[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')

charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

#########################
#correct with lexicon
#correct test with lexicon
lexicon_txt=[]
test_lexicon_txt=[]
#######
lexicon_path='D:/IIIT-HW-Dev/lexicon.txt' 
f_lexicon=open(lexicon_path, encoding="utf8")
for line in f_lexicon:
    #print(line)    
    txt=line.strip() 
    lexicon_txt.append(txt) ; 
    test_lexicon_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
#####                   
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0  
#numCharErrLexicon=0  
num_Lexicon_corrected=0

listtt=[]
tst=[]
#tst=test_txt[:] 
tst= test_lexicon_txt[:]  
#tst=valid_txt[:10]         
print('Ground truth -> Recognized')	
#for i in range(len(listt[:100])):
for i in range(300,400): 
    #print(i)
    if tst[i] == listt[i]:
        numWordOK += 1
        listtt.append(listt[i])
    else:
        # correct predicted listt[] with lexicon
        distance=[]  
        #print(i) 
        for j in range(len(tst)):
            dist = (editdistance.eval(listt[i], tst[j]))
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
        listtt.append(tst[res[0]])
        num_Lexicon_corrected+=1            
                         
    numWordTotal += 1   
    numCharErr+=dist
    numCharTotal += len(tst[i])
#calculate results before lexicon correction 
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

      
#calculate results after lexicon correction 
# now predicted listt is converted into listtt with lexicon correction
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=test_txt[300:400]        
print('Ground truth -> Recognized')	
for i in range(len(listtt)):  # listtt is lexicon corrected
    #print(i)
    if len(listtt[i])==len(tst[i]):        
        if tst[i] == listtt[i]:               
            numWordOK += 1
        else:
            numWordOK += 0
    else:
        numWordOK += 0 
        
    numWordTotal += 1   
    dist = editdistance.eval(listtt[i], tst[i])
    numCharErr+=dist
    numCharTotal += len(tst[i])
    
    if dist==0:
        print('[OK]') 
    else:         
        for p in tst[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')
        for p in listtt[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')


charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
 
#####################    
    
##Beam-Search algo

print("original_text =  ", test_orig_txt[10:11]) #बना
prediction = act_model_Dev.predict(test_img[10:11]) 
top_paths = 3
results = []
for i in range(top_paths):
  lables = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                       greedy=False, beam_width=top_paths, top_paths=top_paths)[0][i])[0]
  results.append(lables)
  

for p in results: 
   for y in p:
      if int(y) != -1:
          print(char_list[int(y)], end = '')       
   print('\n')
#############################################
#check lexicon corrected word in listtt
for p in listtt[12859:12860]: 
   for y in p:
      if int(y) != -1:
          print(char_list[int(y)], end = '')       
   print('\n')   
   
#########################

#decode with Beam-Search algo
prediction = act_model_Dev.predict(test_img[100:200])
#print("original_text =  ", test_orig_txt[:5])

Beam_S_List=[]

#for m in range(len(test_orig_txt[:])):

#for m in range(len(test_orig_txt[:2000])): 

for m in range(10000, 12865):     
    print(m)
    
    prediction = act_model_Dev.predict(test_img[m:m+1])
    top_paths = 3
    results = []
    for i in range(top_paths):
       
        lables = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                       greedy=False, beam_width=top_paths, top_paths=top_paths)[0][i])[0]
        results.append(lables)

    Beam_S_List.append(results)
    
############################################
#correct Beam-Search algo results with lexicon
import editdistance    
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0  
num_Lexicon_corrected=0     
listtt_beam=[]
tst=[]
#tst=test_txt[:100] 
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
    numCharTotal += len(tst[i])
'''    
# select unique items in listtt_beam    
output = []
for x in listtt_beam:
    #print(x)
    if x not in output:
        output.append(x)
#print(output)  '''
        
'''
zz=0
listtt_beam_2=[]
output1 = []
listtt_beam_2 = listtt_beam[zz:zz+3]
for x in listtt_beam_2:
    #print(x)
    if x not in output1:
        output1.append(x) 
       # output.append(output1)     
zz+=3   '''
    
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
#calculate results before lexicon correction 
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

##########################################
# calculate results after lexicon correction in Beam-Search algo
# now predicted output i.e unique(listtt_beam) is evaluated with lexicon correction
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=test_txt[100:200]        
print('Ground truth -> Recognized')	
for i in range(len(output)):  # listtt is lexicon corrected
    #print(i)
    if len(output[i])==len([tst[i]]):        
        if [tst[i]] == output[i]:              
            numWordOK += 1
        else:
            numWordOK += 0
    else:
        numWordOK += 0 
        
    numWordTotal += 1 
    distance=[]  
    for y in range(len(output[i])):
        dist = editdistance.eval(output[i][y], tst[i])
        distance.append(dist)
        temp = min(distance)    
        
        numCharErr+=temp
        numCharTotal += len(tst[i])
    
        if dist==0:
            print('[OK]') 
        else: 
            for p in tst[i]:  
                print(char_list[int(p)], end = '') 
                #print('\n')
            for z in range(len(output[i])):    
                for p in output[i][y]:  
                    print(char_list[int(p)], end = '') 
                    #print('\n')


charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

######################################
#Output list contain unique lexicon corrected Beam-search algo results of trained model on test data.
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

    if len(output[i])==1:
        print("length is one")
        print(i)
        if ([tst[i]] == output[i]):
            numWordOK += 1
            print("TRUE")
        else:
             numWordOK += 0   
        
    elif len(output[i])==2:
        print("length is two")
        print(i)
        if (tst[i] == output[i][0]) or (tst[i] == output[i][1]):
            numWordOK += 1 
            print("TRUE")
        else:
            numWordOK += 0         
            
    else : # if len(output[i])==3:
        print("length is three")
        print(i)
        if (tst[i] == output[i][0]) or (tst[i] == output[i][1]) or (tst[i] == output[i][2]):
            numWordOK += 1 
            print("TRUE")
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
        numCharTotal += len(tst[i])
    
        if dist==0:
            print('[OK]') 
        else: 
            for p in tst[i]:  
                print(char_list[int(p)], end = '') 
                #print('\n')
            for z in range(len(output[i])):    
                for p in output[i][y]:  
                    print(char_list[int(p)], end = '') 
                    #print('\n')


charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

 
###################################################
#Plotting

import matplotlib.pyplot as plt
# Accuracy
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

   
#################################################################################################################   
#################################################################################################################   

# testing on Indicword Dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

path = 'D:/IndicWord/a03'
f=open(path+'/'+'a03.txt', encoding='utf-8-sig')
for line in f:
    print(line)    
    txt=line.strip().split(' ')[8]
    valid_orig_txt.append(txt) ; 
    valid_label_length.append(len(str(txt)));
    valid_input_length.append(31); 
    valid_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))]));  
    img_path_=line.split(' ')[0] ; 
    img_path=(path+'/'+img_path_+'.jpg');  
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)      
     #convert each image of shape (32, 128, 1)
    '''(w, h) = img.shape
    if w>32 or h>128:
        #continue
        img = cv2.resize(img,(128,32))
    if w < 32:
        add_zeros = np.ones((32-w, h))*255
        img = np.concatenate((img, add_zeros))
    if h < 128:
        add_zeros = np.ones((32, 128-h))*255
        img = np.concatenate((img, add_zeros), axis=1)
    
    (w, h) = img.shape
    if w!=32 and h!=128:
        img=cv2.resize(img,(128,32))'''
        
    img =cv2.resize(img,(128,32))    
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.
        
    valid_img.append(img)  
  

valid_img = np.array(valid_img)    

prediction = act_model_Dev.predict(valid_img[:])

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
# see the results  

i = 0
for x in out:
    print("original_text =  ", valid_orig_txt[i])#:test_orig_txt[i])
    print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')       
    print('\n')
    i+=1
    #################
#code to make list, subset of another list after droping some elements. 
#in loop 
listt=[]
for i in range(len(out)):
    #print(i)
    lst=[]
    for x in out[i]:
        #print(x)
        if x!=-1:
            lst.append(x)
    listt.append(lst)   

  ###########################
# without lexicon correction
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=valid_txt[:]
#tst=valid_txt[:10]        
print('Ground truth -> Recognized')	
for i in range(len(listt)):
    #print(i)
    if len(listt[i])==len(tst[i]):        
        if tst[i] == listt[i]:
            numWordOK += 1
        else:
            numWordOK += 0
    else:
        numWordOK += 0 
        
    numWordTotal += 1   
    dist = editdistance.eval(listt[i], tst[i])
    numCharErr+=dist
    numCharTotal += len(tst[i])
    
    if dist==0:
        print('[OK]') 
    else: 
        for p in tst[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')
        for p in listt[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')

charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))

 ######################
#correct with lexicon
#correct test with lexicon

lexicon_txt=[]
test_lexicon_txt=[]

lexicon_path='D:/IndicWord/LexiconHindi.txt' 
f_lexicon=open(lexicon_path, encoding="utf8")
for line in f_lexicon:
    #print(line)    
    txt=line.strip() 
    lexicon_txt.append(txt) ; 
    test_lexicon_txt.append(encode_to_labels(list(str(txt))[0:len(str(txt))])); 
                   
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0  
#numCharErrLexicon=0  
num_Lexicon_corrected=0

listtt=[]
tst=[]
tst=valid_txt[:]  
#tst=valid_txt[:10]         
print('Ground truth -> Recognized')	
for i in range(len(listt)):
    if tst[i] == listt[i]:
        numWordOK += 1
        listtt.append(listt[i])
    else:
        # correct predicted listt[] with lexicon
        distance=[]  
        #print(i) 
        for j in range(len(tst)):
            dist = (editdistance.eval(listt[i], tst[j]))
            distance.append(dist)
            temp = min(distance) 
            #find index of minimun dist
            res = [] 
            for idx in range(0, len(distance)): 
                if temp == distance[idx]: 
                    res.append(idx)
                   # print("The Positions of minimum element : " + str(res)) 
                    #The Positions of minimum element : [6101, 6267, 8583]        
        #to do: if res[] has more than 1 element code for lexicon correction
        listtt.append(tst[res[0]])
        num_Lexicon_corrected+=1            
                         
    numWordTotal += 1   
    numCharErr+=dist
    numCharTotal += len(tst[i])
#calculate results before lexicon correction 
charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
      
#calculate results after lexicon correction 
# now predicted listt is converted into listtt with lexicon correction
import editdistance
numCharErr = 0
numCharTotal = 0
numWordOK = 0
numWordTotal = 0            

tst=[]
tst=valid_txt[:]        
print('Ground truth -> Recognized')	
for i in range(len(listtt)):  # listtt is lexicon corrected
    #print(i)
    if len(listtt[i])==len(tst[i]):        
        if tst[i] == listtt[i]:
            numWordOK += 1
        else:
            numWordOK += 0
    else:
        numWordOK += 0 
        
    numWordTotal += 1   
    dist = editdistance.eval(listtt[i], tst[i])
    numCharErr+=dist
    numCharTotal += len(tst[i])
    
    if dist==0:
        print('[OK]') 
    else: 
        for p in tst[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')
        for p in listtt[i]:  
            print(char_list[int(p)], end = '') 
        print('\n')


charErrorRate = numCharErr / numCharTotal
wordAccuracy = numWordOK / numWordTotal
print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
       
     
######################################
