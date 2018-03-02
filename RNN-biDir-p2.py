from __future__ import division
from __future__ import absolute_import
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
class basicRNN(object):
    
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
        self.hls_infer = np.zeros((lenRec,sizeHidden))
        self.hrs_infer = np.zeros((lenRec,sizeHidden))
        
        self.W  = np.zeros((lenOut,sizeHidden*2)) # for the last fully connected layer
        self.b  = np.zeros(lenOut)
       
        # for training phase 
        self.xs = np.zeros((lenRec,lenIn))
        self.ys = np.zeros((lenRec,lenOut))
        self.hls = np.zeros((lenRec,sizeHidden))
        self.hrs = np.zeros((lenRec,sizeHidden))
        self.GW = np.zeros((lenOut,sizeHidden*2)) # Gradient, for W-update using RMSprop
        self.Gb = np.zeros(lenOut)
        
        # CELL class
        self.RNN_cell = RNN_cell(sizeHidden+lenIn,sizeHidden,lenRec,learningRate)
        
        u''' end of basicRNN.__init__ '''
       
    u''' This is used when mini-batch is used '''            
    def update_inputs_targets(self, inputs_encoded, targets):
        self.inputs_encoded  = inputs_encoded
        self.targets         = targets
    
    def fwd_pass(self): 
        # fwd layer
        prev_h = np.zeros_like(self.hls[0])
        for t in xrange(0,self.lenRec):
            # update input
            self.x    = self.inputs_encoded[t]
            self.xs[t]= self.inputs_encoded[t]
            
            self.RNN_cell.hlx = np.hstack((prev_h, self.x));
           
            hl = self.RNN_cell.fwd_pass_L()
            # bookkeeping
            self.hls[t] = hl
            prev_h = self.hls[t]
                           
        # bwd layer
        prev_h = np.zeros_like(self.hrs[0])                 
        for t in reversed(xrange(0,self.lenRec)):
            # update input
            self.x    = self.xs[t]
            self.RNN_cell.hrx = np.hstack((prev_h, self.x));
           
            hr = self.RNN_cell.fwd_pass_R()
            # bookkeeping
            self.hrs[t] = hr
            prev_h = self.hrs[t] 
                           
            # output layer - fully connected layer
            self.ys[t] = np.dot(self.W,np.hstack((self.hls[t],self.hrs[t]))) + self.b            
        return;              
    
    def bwd_pass(self):        

        avg_loss = 0; # using cross entropy average
        h2next_grad  = np.zeros(self.sizeHidden)
        
        # output bp
        W_grad   = np.zeros((self.lenOut,self.sizeHidden*2))
        b_grad  = np.zeros(self.lenOut)
                                
        hlxW_grad  = np.zeros((self.sizeHidden,self.RNN_cell.lenIn));
        hrxW_grad  = np.zeros((self.sizeHidden,self.RNN_cell.lenIn));                        
        hlb_grad   = np.zeros((self.sizeHidden));
        hrb_grad   = np.zeros((self.sizeHidden)); 
                                
        # propagates through time and layers      
        dh = np.zeros((lenRec,sizeHidden*2))                

        for t in reversed(xrange(0,self.lenRec)):
            
            prob = softmax(self.ys[t]) # prevent zero
            prob_fix  = prob + 1e-9

            # cross entropy
            err       = np.log(prob_fix[self.targets[t]])
            avg_loss += err
     
            dy = copy.deepcopy(prob)
            dy[self.targets[t]] -= 1
            
            W_grad += np.dot((np.atleast_2d(dy)).T,np.atleast_2d(np.hstack((self.hls[t],self.hrs[t])) ))
            b_grad += dy
            
            dh[t] = np.dot(self.W.T,dy) 
                                
        for t in reversed(xrange(0,self.lenRec)):                 
            dhl = dh[t,:self.sizeHidden] + h2next_grad         
            x_grad  = np.zeros(self.lenIn)
            
            if(t > 0):
                prev_h = self.hls[t-1]
            else:
                prev_h = np.zeros_like(self.hls[0])
                
            self.RNN_cell.hlx = np.hstack((prev_h,self.xs[t]))
            self.RNN_cell.hl  = self.hls[t]

            dhlxW, dhlb, h2next_grad,x_grad = \
            self.RNN_cell.bwd_pass_L( dhl );
            
            hlxW_grad  +=  dhlxW
            hlb_grad   +=  dhlb
                                
        h2next_grad  = np.zeros(self.sizeHidden)                        
        for t in xrange(0,self.lenRec):                 
            dhr = dh[t,self.sizeHidden:] + h2next_grad         
            x_grad  = np.zeros(self.lenIn)
            
            if(t < self.lenRec-1):
                prev_h = self.hrs[t+1]
            else:
                prev_h = np.zeros_like(self.hrs[0])
                
            self.RNN_cell.hrx = np.hstack((prev_h,self.xs[t]))
            self.RNN_cell.hr  = self.hrs[t]

            dhrxW, dhrb, h2next_grad,x_grad = \
            self.RNN_cell.bwd_pass_R( dhr );
            
            hrxW_grad  +=  dhrxW
            hrb_grad   +=  dhrb
                                
        self.RNN_cell.update(hlxW_grad/self.lenRec, hlb_grad/self.lenRec,hrxW_grad/self.lenRec, hrb_grad/self.lenRec);
        
        self.update(W_grad/self.lenRec,b_grad/self.lenRec);
        return avg_loss/self.lenRec;
            
    def update(self, W_grad, b_grad):
        self.GW = self.GW + W_grad**2;
        self.W -= self.learningRate/np.sqrt(self.GW + 1e-8) * W_grad;
        self.Gb = self.Gb + b_grad**2;
        self.b -= self.learningRate/np.sqrt(self.Gb + 1e-8) * b_grad;

    def inference(self,xs):
        # fwd layer
        prev_h = np.zeros_like(self.hls_infer[0])
        for t in xrange(0,self.lenRec):
            # update input
            self.x    = xs[t]
            
            self.RNN_cell.hlx = np.hstack((prev_h, self.x));
           
            hl = self.RNN_cell.fwd_pass_L()
            # bookkeeping
            self.hls_infer[t] = hl
            prev_h = self.hls_infer[t]
                           
        # bwd layer
        prev_h = np.zeros_like(self.hrs[0])                 
        for t in reversed(xrange(0,self.lenRec)):
            # update input
            self.x    = xs[t]
            self.RNN_cell.hrx = np.hstack((prev_h, self.x));
           
            hr = self.RNN_cell.fwd_pass_R()
            # bookkeeping
            self.hrs_infer[t] = hr
            prev_h = self.hrs_infer[t] 
                           
            # output layer - fully connected layer
        y = np.dot(self.W,np.hstack((self.hls_infer[0],self.hrs_infer[0]))) + self.b 
        p = softmax(y)
             
        return np.random.choice(xrange(self.lenIn), p=p.ravel())

class RNN_cell(object):
    
    def __init__ (self,lenIn,sizeHidden,lenRec,learningRate):
        self.lenIn        = lenIn
        self.sizeHidden   = sizeHidden
        self.lenRec       = lenRec
        self.learningRate = learningRate
        
        # hx == x is x and h horizontally stacked together
        self.hlx = np.zeros(lenIn)
        self.hrx = np.zeros(lenIn)
        self.hl = np.zeros(sizeHidden)
        self.hr = np.zeros(sizeHidden)
        
        # Weight matrices
        self.hlxW = np.random.random((sizeHidden,lenIn));
        self.hrxW = np.random.random((sizeHidden,lenIn));
        
        # biases
        self.hlb = np.zeros(sizeHidden);
        self.hrb = np.zeros(sizeHidden);
        
        # for RMSprop only
        self.GhlxW = np.random.random((sizeHidden,lenIn));
        self.GhrxW = np.random.random((sizeHidden,lenIn));
        self.Ghlb = np.zeros(sizeHidden);
        self.Ghrb = np.zeros(sizeHidden);
        
        u''' end of RNN_cell.__init__ '''
        
    def fwd_pass_L(self):
        self.hl = tanh(np.dot(self.hlxW, self.hlx) + self.hlb)       
        return self.hl;

    def fwd_pass_R(self):
        self.hr = tanh(np.dot(self.hrxW, self.hrx) + self.hrb)       
        return self.hr;

    def bwd_pass_L(self, dhl):
        
        dhl = np.clip(dhl, -6, 6);       
        # h = o*tanh(c)
        dhl  = tanh_deriv(self.hl) * dhl
        dhlb = dhl
        dhlxW = np.dot((np.atleast_2d(dhl)).T,np.atleast_2d(self.hlx)) 
        
        hlx_grad = np.dot(self.hlxW.T, dhl)
               
        return dhlxW, dhlb, hlx_grad[:self.sizeHidden],hlx_grad[self.sizeHidden:];

    def bwd_pass_R(self, dhr):
        
        dhr = np.clip(dhr, -6, 6);       
        # h = o*tanh(c)
        dhr  = tanh_deriv(self.hr) * dhr
        dhrb = dhr
        dhrxW = np.dot((np.atleast_2d(dhr)).T,np.atleast_2d(self.hrx)) 
        
        hrx_grad = np.dot(self.hrxW.T, dhr)
               
        return dhrxW, dhrb, hrx_grad[:self.sizeHidden],hrx_grad[self.sizeHidden:];
    
    def update(self, hlxW_grad, hlb_grad, hrxW_grad, hrb_grad):

        # adagrad
        self.GhlxW = self.GhlxW + hlxW_grad**2
        self.Ghlb = self.Ghlb + hlb_grad**2
        self.GhrxW = self.GhrxW + hrxW_grad**2
        self.Ghrb = self.Ghrb + hrb_grad**2
        
        self.hlxW -= self.learningRate/np.sqrt(self.GhlxW + 1e-8) * hlxW_grad
        self.hlb -= self.learningRate/np.sqrt(self.Ghlb + 1e-8) * hlb_grad
        self.hrxW -= self.learningRate/np.sqrt(self.GhrxW + 1e-8) * hrxW_grad
        self.hrb -= self.learningRate/np.sqrt(self.Ghrb + 1e-8) * hrb_grad
