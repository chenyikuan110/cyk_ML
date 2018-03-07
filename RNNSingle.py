
# coding: utf-8

# In[1]:

import numpy as np
import copy

# Helper functions
def softmax(array):
    return np.exp(array)/ np.sum(np.exp(array)) # return an array

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_deriv(y):
    return (y*(1-y))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(y):
    return 1 - pow(np.tanh(y),2)

# RNN
class basicRNN:
    
    def __init__ (self, lenIn, lenOut, lenRec, sizeHidden, inputs_encoded, targets, learningRate):
        
        # Hyper parameters
        self.lenIn          = lenIn
        self.lenOut         = lenOut
        self.lenRec         = lenRec
        self.sizeHidden     = sizeHidden
        self.learningRate   = learningRate
        
        # input & expected output
        self.inputs_encoded = inputs_encoded;
        self.targets = targets;
        
        # parameters for inference
        self.x  = np.zeros(lenIn)  
        self.y  = np.zeros(lenOut)
        self.h  = np.zeros(sizeHidden)
        
        self.W  = np.zeros((lenOut,sizeHidden)) # for the last fully connected layer        
        self.b  = np.zeros(lenOut)
        
        
        # for training phase 
        self.xs = np.zeros((lenRec,lenIn))
        self.ys = np.zeros((lenRec,lenOut))
        self.hs = np.zeros((lenRec,sizeHidden))
        self.GW = np.zeros((lenOut,sizeHidden)) # Gradient, for W-update using RMSprop
        self.Gb = np.zeros(lenOut)
        
        # CELL class
        self.RNN_cell = RNN_cell(sizeHidden+lenIn,sizeHidden,lenRec,learningRate)
        
        ''' end of basicRNN.__init__ '''
       
    ''' This is used when mini-batch is used '''            
    def update_inputs_targets(self, inputs_encoded, targets):
        self.inputs_encoded  = inputs_encoded
        self.targets         = targets
    
    def fwd_pass(self): 
        prev_h = np.zeros_like(self.hs[0])
        for t in range(0,self.lenRec):
            # update input
            self.x    = self.inputs_encoded[t]
            self.xs[t]= self.inputs_encoded[t]
            
            self.RNN_cell.hx = np.hstack((prev_h, self.x));
           
            h = self.RNN_cell.fwd_pass()
            # bookkeeping
            self.hs[t] = h
            
            # output layer - fully connected layer
            self.ys[t] = np.dot(self.W,self.hs[t]) + self.b
            prev_h = self.hs[t]
            
        return;              
    
    def bwd_pass(self):        

        avg_loss = 0; # using cross entropy average
        h2next_grad  = np.zeros(self.sizeHidden)
        
        # output bp
        W_grad   = np.zeros((self.lenOut,self.sizeHidden))
        b_grad  = np.zeros(self.lenOut)
        hxW_grad  = np.zeros((self.sizeHidden,self.RNN_cell.lenIn));
        hb_grad   = np.zeros((self.sizeHidden));
                   
        # propagates through time and layers

        for t in reversed(range(0,self.lenRec)):
            
            prob = softmax(self.ys[t]) # prevent zero
            prob_fix  = prob + 1e-9

            # cross entropy
            err       = np.log(prob_fix[int(self.targets[t])])
            avg_loss += err
     
            dy = copy.deepcopy(prob)
            dy[int(self.targets[t])] -= 1
            
            W_grad += np.dot((np.atleast_2d(dy)).T,np.atleast_2d(self.hs[t]))
            b_grad += dy
            
            dh = np.dot(self.W.T,dy) + h2next_grad
            
            x_grad  = np.zeros(self.lenIn)
            
            if(t > 0):
                prev_h = self.hs[t-1]
            else:
                prev_h = np.zeros_like(self.hs[0])
                
            self.RNN_cell.hx = np.hstack((prev_h,self.xs[t]))
            self.RNN_cell.h  = self.hs[t]

            dhxW, dhb, h2next_grad,x_grad =             self.RNN_cell.bwd_pass( dh );
            
            hxW_grad  +=  dhxW
            hb_grad   +=  dhb
            
        self.RNN_cell.update(hxW_grad/self.lenRec, hb_grad/self.lenRec);
        
        self.update(W_grad/self.lenRec,b_grad/self.lenRec);
        return avg_loss/self.lenRec;
            
          
            
    def update(self, W_grad, b_grad):
        self.GW = self.GW + W_grad**2;
        self.W -= self.learningRate/np.sqrt(self.GW + 1e-8) * W_grad;
        self.Gb = self.Gb + b_grad**2;
        self.b -= self.learningRate/np.sqrt(self.Gb + 1e-8) * b_grad;

    def inference(self,x):
        # update input
        self.x = x
        self.RNN_cell.hx = np.hstack((self.h,self.x))
        self.h = self.RNN_cell.fwd_pass()

        # output layer - may replace with softmax instead
        self.y = np.dot(self.W,self.h) + self.b
        p   = softmax(self.y)
        
        
        return np.random.choice(range(self.lenOut), p=p.ravel())
  
    def get_prob(self,x):
        # update input
        self.x = x
        self.RNN_cell.hx = np.hstack((self.h,self.x))
        self.h = self.RNN_cell.fwd_pass()

        # output layer - may replace with softmax instead
        self.y = np.dot(self.W,self.h) + self.b
        p   = softmax(self.y)
        
        
        return p[1];


# In[32]:

class RNN_cell:
    
    def __init__ (self,lenIn,sizeHidden,lenRec,learningRate):
        self.lenIn        = lenIn
        self.sizeHidden   = sizeHidden
        self.lenRec       = lenRec
        self.learningRate = learningRate
        
        # hx == x is x and h horizontally stacked together
        self.hx = np.zeros(lenIn)
        self.h = np.zeros(sizeHidden)
        
        # Weight matrices
        self.hxW = np.random.random((sizeHidden,lenIn));

        # biases
        self.hb = np.zeros(sizeHidden);
              
        # for RMSprop only
        self.GhxW = np.random.random((sizeHidden,lenIn));
        self.Ghb = np.zeros(sizeHidden);

        
        ''' end of RNN_cell.__init__ '''
        
    def fwd_pass(self):
        self.h = tanh(np.dot(self.hxW, self.hx) + self.hb)       
        return self.h;
    
    def bwd_pass(self, dh):
        
        dh = np.clip(dh, -6, 6);       
        # h = o*tanh(c)
        dh  = tanh_deriv(self.h) * dh
        dhb = dh
        dhxW = np.dot((np.atleast_2d(dh)).T,np.atleast_2d(self.hx)) 
        
        hx_grad = np.dot(self.hxW.T, dh)
               
        return dhxW, dhb, hx_grad[:self.sizeHidden],hx_grad[self.sizeHidden:];
    
    def update(self, hxW_grad, hb_grad):

        self.GhxW = self.GhxW + hxW_grad**2
        self.Ghb = self.Ghb + hb_grad**2
        
        self.hxW -= self.learningRate/np.sqrt(self.GhxW + 1e-8) * hxW_grad
        self.hb -= self.learningRate/np.sqrt(self.Ghb + 1e-8) * hb_grad

        

def encode(idx,num_entry):
    ret = np.zeros(num_entry)
    ret[idx] = 1
    return ret;

def encode_array(array,num_entry):
    xs = np.zeros((len(array),num_entry))
    for i in range(len(array)):
        xs[i][array[i]] = 1; 
    return xs;




