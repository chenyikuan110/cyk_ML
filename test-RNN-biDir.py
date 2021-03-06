import numpy as np
import copy


data = open('HP1.txt','r', encoding="utf8").read();
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(data_size,", ",vocab_size)

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)

def encode(idx,num_entry):
    ret = np.zeros(num_entry)
    ret[idx] = 1
    return ret;

def encode_array(array,num_entry):
    xs = np.zeros((len(array),num_entry))
    for i in range(len(array)):
        xs[i][array[i]] = 1; 
    return xs;


seq_length,position = 100,0
inputs = [char_to_ix[ch] for ch in data[position:position+seq_length]]
print(data[position:position+seq_length])
print("inputs",inputs)

targets = [char_to_ix[ch] for ch in data[position+1:position+seq_length+1]] 
print(data[position+1:position+seq_length+1])
print("targets",targets)

n,position = 0,0;
epoch = 30*1000;
lenIn, lenOut, lenRec = vocab_size,vocab_size, seq_length
sizeHidden, numHiddenLayer = 100,1;
learningRate = 0.1;


R = basicRNN(lenIn, lenOut, lenRec, sizeHidden, encode_array(inputs,vocab_size),targets, learningRate)

# training
while n<epoch:
    
    if(position+seq_length+1 >= len(data) or n == 0):
        print("!!!!",len(data))
        position = 0;
        
    inputs  = [char_to_ix[ch] for ch in data[position:position+seq_length]]
    targets = [char_to_ix[ch] for ch in data[position+1:position+seq_length+1]] 

    R.update_inputs_targets(encode_array(inputs,vocab_size),targets)
    R.fwd_pass();
    
    err = R.bwd_pass();
    
    if(n%500 == 0):
        print(n,"err:",err)
        infer_in  = [char_to_ix[ch] for ch in data[position:position+seq_length]]
        infer_in_enc = encode_array(infer_in,vocab_size)
        result = [];

        for i in range(200):
            ret = R.inference(infer_in_enc)
            #print(i,":",ret)
            result.append(ret)
            infer_in.append(ret)
            infer_in_enc = encode_array(infer_in[i+1:],vocab_size)
        decode = ''.join([ix_to_char[ch] for ch in result] )
        print(decode+'\n')

    position += seq_length;
    n += 1;