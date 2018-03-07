
# coding: utf-8

# In[18]:

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
class lstmRNN:
    
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
        self.cls_infer = np.zeros((lenRec,sizeHidden))
        self.crs_infer = np.zeros((lenRec,sizeHidden))
        self.W  = np.zeros((lenOut,sizeHidden*2)) # for the last fully connected layer
        self.b  = np.zeros(lenOut)
       
        # for training phase 
        self.xs = np.zeros((lenRec,lenIn))
        self.ys = np.zeros((lenRec,lenOut))
        self.hls = np.zeros((lenRec,sizeHidden))
        self.hrs = np.zeros((lenRec,sizeHidden))
        self.cls = np.zeros((lenRec,sizeHidden))
        self.crs = np.zeros((lenRec,sizeHidden))
        self.GW = np.zeros((lenOut,sizeHidden*2)) # Gradient, for W-update using RMSprop
        self.Gb = np.zeros(lenOut)
        
        # for training phase bookkeeping
        self.flg = np.zeros((lenRec,sizeHidden)) # forget gate
        self.frg = np.zeros((lenRec,sizeHidden))
        self.ilg = np.zeros((lenRec,sizeHidden)) # input  gate
        self.irg = np.zeros((lenRec,sizeHidden))
        self.olg = np.zeros((lenRec,sizeHidden)) # output gate
        self.org = np.zeros((lenRec,sizeHidden))
        self.mlc = np.zeros((lenRec,sizeHidden)) # memory cell
        self.mrc = np.zeros((lenRec,sizeHidden))
        
        # LSTM class
        self.LSTM_L = LSTM(sizeHidden+lenIn,sizeHidden,lenRec,learningRate)
        self.LSTM_R = LSTM(sizeHidden+lenIn,sizeHidden,lenRec,learningRate)
        
        ''' end of lstmRNN.__init__ '''
       
    ''' This is used when mini-batch is used '''            
    def update_inputs_targets(self, inputs_encoded, targets):
        self.inputs_encoded  = inputs_encoded
        self.targets         = targets
    
    def fwd_pass(self): 
        # fwd layer
        prev_h = np.zeros_like(self.hls[0])
        for t in range(0,self.lenRec):
            # update input - edited until here
            self.x    = self.inputs_encoded[t]
            self.xs[t]= self.inputs_encoded[t]
            
            self.LSTM_L.hx = np.hstack((prev_h, self.x));
           
            c,h,f,i,m,o = self.LSTM_L.fwd_pass()
            # bookkeeping
            self.cls[t] = c
            self.hls[t] = h
            self.flg[t] = f
            self.ilg[t] = i
            self.mlc[t] = m
            self.olg[t] = o
            prev_h = self.hls[t]
                           
        # bwd layer
        prev_h = np.zeros_like(self.hrs[0])                 
        for t in reversed(range(0,self.lenRec)):
            # update input
            self.x    = self.xs[t]
            self.LSTM_R.hrx = np.hstack((prev_h, self.x));
           
            c,h,f,i,m,o = self.LSTM_R.fwd_pass()
            # bookkeeping
            self.crs[t] = c
            self.hrs[t] = h
            self.frg[t] = f
            self.irg[t] = i
            self.mrc[t] = m
            self.org[t] = o
            prev_h = self.hrs[t] 
                           
            # output layer - fully connected layer
            self.ys[t] = np.dot(self.W,np.hstack((self.hls[t],self.hrs[t]))) + self.b            
        return;              
    
    def bwd_pass(self):        

        avg_loss = 0; # using cross entropy average
        h2next_grad  = np.zeros(self.sizeHidden)
        c2next_grad  = np.zeros(self.sizeHidden)
        
        # output bp
        W_grad   = np.zeros((self.lenOut,self.sizeHidden*2))
        b_grad  = np.zeros(self.lenOut)
                                
        hlxf_grad  = np.zeros((self.sizeHidden,self.LSTM_L.lenIn));
        hrxf_grad  = np.zeros((self.sizeHidden,self.LSTM_R.lenIn));   
        hlxi_grad  = np.zeros((self.sizeHidden,self.LSTM_L.lenIn));
        hrxi_grad  = np.zeros((self.sizeHidden,self.LSTM_R.lenIn));
        hlxm_grad  = np.zeros((self.sizeHidden,self.LSTM_L.lenIn));
        hrxm_grad  = np.zeros((self.sizeHidden,self.LSTM_R.lenIn));
        hlxo_grad  = np.zeros((self.sizeHidden,self.LSTM_L.lenIn));
        hrxo_grad  = np.zeros((self.sizeHidden,self.LSTM_R.lenIn));

        flb_grad   = np.zeros((self.sizeHidden));
        frb_grad   = np.zeros((self.sizeHidden)); 
        ilb_grad   = np.zeros((self.sizeHidden));
        irb_grad   = np.zeros((self.sizeHidden)); 
        mlb_grad   = np.zeros((self.sizeHidden));
        mrb_grad   = np.zeros((self.sizeHidden)); 
        olb_grad   = np.zeros((self.sizeHidden));
        orb_grad   = np.zeros((self.sizeHidden)); 
                                
        # propagates through time and layers      
        dh = np.zeros((self.lenRec,self.sizeHidden*2))                

        for t in reversed(range(0,self.lenRec)):
            
            prob = softmax(self.ys[t]) # prevent zero
            prob_fix  = prob + 1e-9

            # cross entropy
            err       = np.log(prob_fix[int(self.targets[t])])
            avg_loss += err
     
            dy = copy.deepcopy(prob)
            dy[int(self.targets[t])] -= 1
            
            W_grad += np.dot((np.atleast_2d(dy)).T,np.atleast_2d(np.hstack((self.hls[t],self.hrs[t])) ))
            b_grad += dy
            
            dh[t] = np.dot(self.W.T,dy) 
                                
        for t in reversed(range(0,self.lenRec)):                 
            dhl = dh[t,:self.sizeHidden] + h2next_grad         
            x_grad  = np.zeros(self.lenIn)
            
            if(t > 0):
                prev_h,prev_c = self.hls[t-1],self.cls[t-1]
            else:
                prev_h,prev_c = np.zeros_like(self.hls[0]),np.zeros_like(self.cls[0])
                
            self.LSTM_L.hx = np.hstack((prev_h,self.xs[t]))
            self.LSTM_L.c  = self.cls[t]

            dhlxf, dhlxi, dhlxm, dhlxo, dblf, dbli, dblm, dblo,c2next_grad, h2next_grad,x_grad =             self.LSTM_L.bwd_pass( dhl, prev_c, self.flg[t],self.ilg[t],self.mlc[t],self.olg[t], c2next_grad);
            
            hlxf_grad  +=  dhlxf
            hlxi_grad  +=  dhlxi
            hlxm_grad  +=  dhlxm
            hlxo_grad  +=  dhlxo         
            flb_grad   +=  dblf
            ilb_grad   +=  dbli
            mlb_grad   +=  dblm
            olb_grad   +=  dblo
                                
        h2next_grad  = np.zeros(self.sizeHidden)     
        c2next_grad  = np.zeros(self.sizeHidden)
        for t in range(0,self.lenRec):                 
            dhr = dh[t,self.sizeHidden:] + h2next_grad         
            x_grad  = np.zeros(self.lenIn)
            
            if(t < self.lenRec-1):
                prev_h,prev_c = self.hrs[t+1],self.crs[t+1]
            else:
                prev_h,prev_c = np.zeros_like(self.hrs[0]),np.zeros_like(self.crs[0])
                
            self.LSTM_R.hx = np.hstack((prev_h,self.xs[t]))
            self.LSTM_R.c  = self.crs[t]

            dhrxf, dhrxi, dhrxm, dhrxo, dbrf, dbri, dbrm, dbro,c2next_grad, h2next_grad,x_grad =             self.LSTM_R.bwd_pass( dhr, prev_c, self.frg[t],self.irg[t],self.mrc[t],self.org[t], c2next_grad);
            
            hrxf_grad  +=  dhrxf
            hrxi_grad  +=  dhrxi
            hrxm_grad  +=  dhrxm
            hrxo_grad  +=  dhrxo         
            frb_grad   +=  dbrf
            irb_grad   +=  dbri
            mrb_grad   +=  dbrm
            orb_grad   +=  dbro
                                
        self.LSTM_L.update(hlxf_grad/self.lenRec, hlxi_grad/self.lenRec,                           hlxm_grad/self.lenRec, hlxo_grad/self.lenRec,                            flb_grad /self.lenRec, ilb_grad /self.lenRec,                            mlb_grad /self.lenRec, olb_grad /self.lenRec)
        self.LSTM_R.update(hrxf_grad/self.lenRec, hrxi_grad/self.lenRec,                           hrxm_grad/self.lenRec, hrxo_grad/self.lenRec,                            frb_grad /self.lenRec, irb_grad /self.lenRec,                            mrb_grad /self.lenRec, orb_grad /self.lenRec)                                          
                         
                                                  
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
        for t in range(0,self.lenRec):
            # update input
            self.x    = xs[t]
            
            self.LSTM_L.hx = np.hstack((prev_h, self.x));
           
            c,h,f,i,m,o = self.LSTM_L.fwd_pass()
            # bookkeeping
            self.hls_infer[t] = h
            self.cls_infer[t] = c
            prev_h = self.hls_infer[t]
           
        # bwd layer
        prev_h = np.zeros_like(self.hrs[0])                 
        for t in reversed(range(0,self.lenRec)):
            # update input
            self.x    = xs[t]
            
            self.LSTM_R.hx = np.hstack((prev_h, self.x));
           
            c,h,f,i,m,o = self.LSTM_R.fwd_pass()
            # bookkeeping
            self.hrs_infer[t] = h
            self.crs_infer[t] = c
            prev_h = self.hrs_infer[t]
                           
            # output layer - fully connected layer
        y = np.dot(self.W,np.hstack((self.hls_infer[self.lenRec-1],self.hrs_infer[self.lenRec-1]))) + self.b 
        p = softmax(y)
             
        return np.random.choice(range(self.lenOut), p=p.ravel())
  


# In[19]:

class LSTM:
    
    def __init__ (self,lenIn,sizeHidden,lenRec,learningRate):
        self.lenIn        = lenIn
        self.sizeHidden   = sizeHidden
        self.lenRec       = lenRec
        self.learningRate = learningRate
        
        # hx == x is x and h horizontally stacked together
        self.hx = np.zeros(lenIn)
        self.c = np.zeros(sizeHidden)
        self.h = np.zeros(sizeHidden)
        
        # Weight matrices
        self.fW = np.random.random((sizeHidden,lenIn));
        self.iW = np.random.random((sizeHidden,lenIn));
        self.mW = np.random.random((sizeHidden,lenIn)); # cell state
        self.oW = np.random.random((sizeHidden,lenIn));
                             
        # biases
        self.fb = np.zeros(sizeHidden);
        self.ib = np.zeros(sizeHidden); 
        self.mb = np.zeros(sizeHidden); 
        self.ob = np.zeros(sizeHidden); 
               
        # for RMSprop only
        self.GfW = np.random.random((sizeHidden,lenIn));
        self.GiW = np.random.random((sizeHidden,lenIn));
        self.GmW = np.random.random((sizeHidden,lenIn)); 
        self.GoW = np.random.random((sizeHidden,lenIn));
                             
        self.Gfb = np.zeros(sizeHidden);
        self.Gib = np.zeros(sizeHidden); 
        self.Gmb = np.zeros(sizeHidden);
        self.Gob = np.zeros(sizeHidden); 
        
        ''' end of LSTM.__init__ '''
        
    def fwd_pass(self):
        f       = sigmoid(np.dot(self.fW, self.hx) + self.fb)
        i       = sigmoid(np.dot(self.iW, self.hx) + self.ib)
        m       = tanh(   np.dot(self.mW, self.hx) + self.mb)        
        o       = sigmoid(np.dot(self.oW, self.hx) + self.ob)
        self.c *= f
        self.c += i * m
        self.h  = o * tanh(self.c)
        
        return self.c, self.h, f, i, m, o;
    
    def bwd_pass(self, dh, prev_c, f, i, m, o, c_g):
        
        dh = np.clip(dh, -6, 6);       
        # h = o*tanh(c)
        do  = tanh(self.c) * dh
        do  = sigmoid_deriv(o)*do
        dhxo = np.dot((np.atleast_2d(do)).T,np.atleast_2d(self.hx)) 
        
        # h = o*tanh(c) - add c_g (c_grad in next timestep, account for the branch here)
        dcs = dh * o * tanh_deriv(self.c) + c_g
        dcs = np.clip(dcs, -6, 6); 
        
        # c = c_prev * f + m * i
        dm = i * dcs
        dm = tanh_deriv(m) * dm
        dhxm = np.dot((np.atleast_2d(dm)).T,np.atleast_2d(self.hx)) 
        
        # c = c_prev * f + m * i
        di  = m * dcs
        di  = sigmoid_deriv(i) * di
        dhxi = np.dot((np.atleast_2d(di)).T,np.atleast_2d(self.hx)) 
        
        # c = c_prev * f + m * i
        df = prev_c * dcs
        df = sigmoid_deriv(f) * df
        dhxf = np.dot((np.atleast_2d(df)).T,np.atleast_2d(self.hx)) 
        
        # c = c_prev * f + m * i
        c_grad  = dcs * f
        hx_grad = np.dot(self.fW.T, df) + np.dot(self.iW.T, di) +                          np.dot(self.oW.T, do) + np.dot(self.mW.T, dm)
        
        
        return dhxf,dhxi,dhxm,dhxo,df,di,dm,do,c_grad,hx_grad[:self.sizeHidden],hx_grad[self.sizeHidden:];
    
    def update(self, f_grad, i_grad, m_grad, o_grad, fb_grad, ib_grad, mb_grad, ob_grad):

        self.GfW = 0.9*self.GfW + 0.1*f_grad**2
        self.GiW = 0.9*self.GiW + 0.1*i_grad**2
        self.GmW = 0.9*self.GmW + 0.1*m_grad**2
        self.GoW = 0.9*self.GoW + 0.1*o_grad**2
        
        self.Gfb = 0.9*self.Gfb + 0.1*fb_grad**2
        self.Gib = 0.9*self.Gib + 0.1*ib_grad**2
        self.Gmb = 0.9*self.Gmb + 0.1*mb_grad**2
        self.Gob = 0.9*self.Gob + 0.1*ob_grad**2
        
        self.fW -= self.learningRate/np.sqrt(self.GfW + 1e-8) * f_grad
        self.iW -= self.learningRate/np.sqrt(self.GiW + 1e-8) * i_grad
        self.mW -= self.learningRate/np.sqrt(self.GmW + 1e-8) * m_grad
        self.oW -= self.learningRate/np.sqrt(self.GoW + 1e-8) * o_grad
        self.fb -= self.learningRate/np.sqrt(self.Gfb + 1e-8) * fb_grad
        self.ib -= self.learningRate/np.sqrt(self.Gib + 1e-8) * ib_grad
        self.mb -= self.learningRate/np.sqrt(self.Gmb + 1e-8) * mb_grad
        self.ob -= self.learningRate/np.sqrt(self.Gob + 1e-8) * ob_grad


def encode(idx,num_entry):
    ret = np.zeros(num_entry)
    ret[idx] = 1
    return ret;

def encode_array(array,num_entry):
    xs = np.zeros((len(array),num_entry))
    for i in range(len(array)):
        xs[i][array[i]] = 1; 
    return xs;



