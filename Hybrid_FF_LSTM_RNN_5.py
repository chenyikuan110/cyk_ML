
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

def shifted_sigmoid(x):
    return (1/(1+np.exp(-x*5+2)))

def shifted_sigmoid_deriv(y):
    return (y*(1-y)*5)


def tanh(x):
    return np.tanh(x)

def tanh_deriv(y):
    return 1 - pow(np.tanh(y),2)

def ReLu(x):
    return 0.5*x + 0.5*(abs(x));

def ReLu_deriv(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if(x[i] > 0):
            y[i] = 1            
    return y

# Partially Recurrent Network - partially-fully dropout recurrent net
# 
# i    h     
# i    h     o
# i          o
# i    h     o
# i    h     o
# i    h     
#
class hFFLSTMRNN:
    
    def __init__ (self, lenIn, lenOut, lenRec, sizeHidden, lenInRec, hiddenRec,\
                  inputs_encoded, targets, \
                  learningRate, dropout_threshold, extra_fc):
        
        # Hyper parameters
        self.lenIn          = lenIn
        self.lenOut         = lenOut
        self.lenRec         = lenRec
        self.sizeHidden     = sizeHidden
        self.lenInRec       = lenInRec
        self.hiddenRec      = hiddenRec
        self.learningRate   = learningRate
        self.dropout_threshold = dropout_threshold
        self.extra_fc         = extra_fc
        
        # input & expected output
        self.inputs_encoded = inputs_encoded;
        self.targets = targets;
        
        # parameters for inference
        self.x  = np.zeros(lenIn)  
        self.y  = np.zeros(lenOut)
        self.prob = np.zeros(lenOut)
        self.is_target = 0;
        
        self.h  = np.zeros(sizeHidden)
        self.h2 = np.zeros(int(sizeHidden/2))
        self.c  = np.zeros(hiddenRec)
        
        self.leaky_W = 0.5
        self.leaky_b = -0.1
        self.Gleaky_W = 0
        self.Gleaky_b = 0
        
        
        if(self.extra_fc == False):
            self.W  = np.zeros((lenOut,sizeHidden)) # for the last fully connected layer 
        else:
            self.W  = np.zeros((lenOut,int(sizeHidden/2)))
                           
        self.b  = np.zeros(lenOut)        
        self.GW = np.zeros_like(self.W) # Gradient, for W-update using RMSprop
        self.Gb = np.zeros_like(self.b)
        
        self.W2  = np.zeros((int(sizeHidden/2),sizeHidden)) # for the second last fully connected layer
        self.b2  = np.zeros(int(sizeHidden/2))
        self.GW2 = np.zeros_like(self.W2)
        self.Gb2 = np.zeros_like(self.b2)
        
        # for training phase 
        self.xs = np.zeros((lenRec,lenIn))       
        self.cs = np.zeros((lenRec,hiddenRec))
        self.hs = np.zeros((lenRec,sizeHidden))
        self.h2s = np.zeros((lenRec,int(sizeHidden/2)))
        self.ys = np.zeros((lenRec,lenOut))
        self.probs = np.zeros((lenRec,lenOut))
        self.is_targets = np.zeros((lenRec,1))
        
        
        # for training phase bookkeeping
        self.fg = np.zeros((lenRec,hiddenRec)) # forget gate
        self.ig = np.zeros((lenRec,hiddenRec)) # input  gate
        self.og = np.zeros((lenRec,hiddenRec)) # output gate
        self.mc = np.zeros((lenRec,hiddenRec)) # memory cell state (candidate)
        
        # LSTM class
        self.LSTM_Rec = LSTM(lenInRec+hiddenRec,hiddenRec,lenRec,learningRate)
        self.FF       = FF(lenIn-lenInRec,sizeHidden-hiddenRec,lenRec,learningRate)
                            
        # Dropout vector
        self.dvo = np.zeros((lenRec,sizeHidden));
        
        ''' end of hFFLSTMRNN.__init__ '''
       
    ''' This is used when mini-batch is used '''            
    def update_inputs_targets(self, inputs_encoded, targets):
        self.inputs_encoded  = inputs_encoded
        self.targets         = targets
    
    def fwd_pass(self): 
                
        prev_h = np.zeros_like(self.hs[0])
        for t in range(0,self.lenRec):
            for i in range(self.dvo.shape[1]):
                rand = np.random.random()
                if(rand > self.dropout_threshold):
                    self.dvo[t][i] = 1;
                else:
                    self.dvo[t][i] = 0;
                    
            # update input
            self.x    = self.inputs_encoded[t]
            self.xs[t]= self.inputs_encoded[t]
            
            # Recurrent part
            self.LSTM_Rec.hx = np.hstack((prev_h[:self.hiddenRec], self.x[:self.lenInRec]));
            self.LSTM_Rec.dvo = self.dvo[t][:self.hiddenRec];     
            cR, hR, fR, iR, mR, oR = self.LSTM_Rec.fwd_pass()            
            self.cs[t] = cR
            self.hs[t][:self.hiddenRec] = hR
            self.fg[t] = fR
            self.ig[t] = iR
            self.mc[t] = mR
            self.og[t] = oR
            
            # Feed forward part 
            self.FF.x  = self.x[self.lenInRec:];
            self.FF.dvo = self.dvo[t][self.hiddenRec:];     
            hN = self.FF.fwd_pass()            
            self.hs[t][self.hiddenRec:] = hN
            prev_h = self.hs[t]
            
            # additional Feed foward fully connected layer - if used
            if(self.extra_fc == True):
                self.h2s[t] = tanh(np.dot(self.W2,self.hs[t]) + self.b2)
                self.ys[t] = np.dot(self.W,self.h2s[t]) + self.b
                self.probs[t] = softmax(self.ys[t]) 
                self.is_targets[t] = shifted_sigmoid(self.is_targets[t-1] + self.leaky_W * self.probs[t][1] - self.leaky_b)
                self.is_targets[t] = np.clip(self.is_targets[t], 0, 1);    
            else:
                # output layer - fully connected layer
                self.ys[t] = np.dot(self.W,self.hs[t]) + self.b
                self.probs[t] = softmax(self.ys[t])             
                self.is_targets[t] = shifted_sigmoid(self.is_targets[t-1] + self.leaky_W * self.probs[t][1] - self.leaky_b)
                self.is_targets[t] = np.clip(self.is_targets[t], 0, 1);
        return;              
    
    def bwd_pass(self):        

        avg_loss = 0; # using cross entropy average
        c2next_grad  = np.zeros(self.hiddenRec)
        h2next_grad  = np.zeros(self.sizeHidden)
        
        # leaky cell bp
        leaky_W_grad   = 0
        leaky_b_grad   = 0
        
        # output bp
        W_grad   = np.zeros_like(self.W)
        b_grad   = np.zeros_like(self.b)
        
        if(self.extra_fc == 1):
            W2_grad  = np.zeros_like(self.W2)
            b2_grad  = np.zeros_like(self.b2)
        
        # LSTM internal bp
        hxf_Rec_grad   = np.zeros((self.hiddenRec,self.LSTM_Rec.lenIn));
        hxi_Rec_grad   = np.zeros((self.hiddenRec,self.LSTM_Rec.lenIn));
        hxm_Rec_grad   = np.zeros((self.hiddenRec,self.LSTM_Rec.lenIn));
        hxo_Rec_grad   = np.zeros((self.hiddenRec,self.LSTM_Rec.lenIn));
        
        hW_grad   = np.zeros((self.sizeHidden-self.hiddenRec,self.FF.lenIn));
        hb_grad   = np.zeros(self.sizeHidden-self.hiddenRec)
        
        fb_grad   = np.zeros(self.hiddenRec)
        ib_grad   = np.zeros(self.hiddenRec)
        mb_grad   = np.zeros(self.hiddenRec)
        ob_grad   = np.zeros(self.hiddenRec)
                   
        # propagates through time and layers

        for t in reversed(range(0,self.lenRec)):
            
            # leaky cell
            # leaky[t] = sigmoid( leaky[t-1] + prob[t][1]*W - b)
            dleaky = self.is_targets[t] - self.targets[t]
            dleaky = shifted_sigmoid_deriv(self.is_targets[t])*dleaky
            
            leaky_W_grad += dleaky * self.probs[t][1]
            leaky_b_grad -= dleaky

            
#            if(prob[self.targets[t]] == 0):
#                for ii in range(10000):
#                    print("ERR!",self.ys[t][self.targets[t]]," ",np.sum(self.ys[t]))
                
            # cross entropy
            # prevent zero
            prob_fix = self.probs[t] + 1e-12
            err       = np.log(prob_fix[int(self.targets[t])])
            avg_loss += err
     
            dy = copy.deepcopy(self.probs[t])
            dy[int(self.targets[t])] -= 1
            
            # if target is 1, but there is a miss, double the cost
            if(self.targets[t] == 1):
                dy *= 2;
            
            if(self.extra_fc == True):
                W_grad += np.dot((np.atleast_2d(dy)).T,np.atleast_2d(self.h2s[t]))
                b_grad += dy
                
                # y = W*h2 + b
                dh2 = np.dot(self.W.T,dy)
                #dh2 = np.clip(dh2, -6, 6);       
                
                # h2 = ReLu(W2*h+b2)
                dh2  = tanh_deriv(self.h2s[t])*dh2              
                dh2W = np.dot((np.atleast_2d(dh2)).T,np.atleast_2d(self.hs[t]))
                            
                dh   = np.dot(self.W2.T,dh2) + h2next_grad
                dh2b = dh2;
                
                W2_grad += dh2W 
                b2_grad += dh2b            
                
            else:
                W_grad += np.dot((np.atleast_2d(dy)).T,np.atleast_2d(self.hs[t]))
                b_grad += dy
            
                dh = np.dot(self.W.T,dy) + h2next_grad

            if(t > 0):
                prev_h = self.hs[t-1]
                prev_c = self.cs[t-1]
            else:
                prev_h = np.zeros_like(self.hs[0])
                prev_c = np.zeros_like(self.cs[0])
                
            # LSTM RNN part
            self.LSTM_Rec.hx = np.hstack((prev_h[:self.hiddenRec],self.xs[t][:self.lenInRec]));
            self.LSTM_Rec.c = self.cs[t];
            
            self.LSTM_Rec.dvo = self.dvo[t][:self.hiddenRec];
            
            dhxf,dhxi,dhxm,dhxo, dbf,dbi,dbm,dbo, c2next_grad,h2next_grad[:self.hiddenRec],x_grad = \
            self.LSTM_Rec.bwd_pass( dh[:self.hiddenRec], prev_c ,self.fg[t],self.ig[t],\
                                   self.mc[t],self.og[t],\
                                 c2next_grad);
            
            for ii in range(dhxo.shape[0]):
                dhxo[ii] *= self.dvo[t][ii];   
                
            hxf_Rec_grad +=  dhxf;
            hxi_Rec_grad +=  dhxi;  
            hxm_Rec_grad +=  dhxm;  
            hxo_Rec_grad +=  dhxo;   
            
            fb_grad[:self.hiddenRec] +=  dbf;
            ib_grad[:self.hiddenRec] +=  dbi;
            mb_grad[:self.hiddenRec] +=  dbm;
            ob_grad[:self.hiddenRec] +=  np.multiply(dbo,self.dvo[t][:self.hiddenRec]);   
            
            # Feed-Forward part
            self.FF.x = self.xs[t][self.lenInRec:];   
            self.FF.h = self.hs[t][self.hiddenRec:]
            dhW_grad, dhb_grad    = self.FF.bwd_pass( dh[self.hiddenRec:] );   
            hW_grad += dhW_grad;
            hb_grad += dhb_grad;

        # update using RMSprop
        self.LSTM_Rec.update(hxf_Rec_grad/self.lenRec, hxi_Rec_grad/self.lenRec, \
                           hxm_Rec_grad/self.lenRec, hxo_Rec_grad/self.lenRec, \
                           fb_grad/self.lenRec, ib_grad/self.lenRec, \
                           mb_grad/self.lenRec, ob_grad/self.lenRec);
        
        self.FF.update(hW_grad/self.lenRec, hb_grad/self.lenRec);
        
        self.update(W_grad/self.lenRec,b_grad/self.lenRec,leaky_W_grad/self.lenRec,leaky_b_grad/self.lenRec );
        
        if(self.extra_fc == True):
            self.update2(W2_grad/self.lenRec,b2_grad/self.lenRec);
        
        return avg_loss/self.lenRec;
            
          
            
    def update(self, W_grad, b_grad, leaky_W_grad, leaky_b_grad):
        self.GW = 0.9*self.GW + 0.1*W_grad**2;
        self.W -= self.learningRate/np.sqrt(self.GW + 1e-8) * W_grad;
        self.Gb = 0.9*self.Gb + 0.1*b_grad**2;
        self.b -= self.learningRate/np.sqrt(self.Gb + 1e-8) * b_grad;
        
        self.Gleaky_W = 0.9*self.Gleaky_W + 0.1*leaky_W_grad**2;
        self.leaky_W -= self.learningRate/np.sqrt(self.Gleaky_W + 1e-8) * leaky_W_grad;
        self.Gleaky_b = 0.9*self.Gleaky_b + 0.1*leaky_b_grad**2;
        self.leaky_b -= self.learningRate/np.sqrt(self.Gleaky_b + 1e-8) * leaky_b_grad;
        
    def update2(self, W2_grad, b2_grad):
        self.GW2 = 0.9*self.GW2 + 0.1*W2_grad**2;
        self.W2 -= self.learningRate/np.sqrt(self.GW2 + 1e-8) * W2_grad;
        self.Gb2 = 0.9*self.Gb2 + 0.1*b2_grad**2;
        self.b2 -= self.learningRate/np.sqrt(self.Gb2 + 1e-8) * b2_grad;
        
    def inference(self,x):
        # update input
        self.x = x
        self.LSTM_Rec.hx = np.hstack((self.h[:self.hiddenRec], self.x[:self.lenInRec]));
        self.LSTM_Rec.dvo = np.ones(self.hiddenRec)   
        cR, hR, fR, iR, mR, oR = self.LSTM_Rec.fwd_pass()            
        self.c = cR
        self.h[:self.hiddenRec] = hR

        # Feed forward part 
        self.FF.x  = self.x[self.lenInRec:];
        self.FF.dvo = np.ones(self.sizeHidden-self.hiddenRec)
        hN = self.FF.fwd_pass()            
        self.h[self.hiddenRec:] = hN

        if(self.extra_fc == True):
            self.h2 = tanh(np.dot(self.W2,self.h) + self.b2)
            self.y = np.dot(self.W,self.h2) + self.b
        else:
            self.y = np.dot(self.W,self.h) + self.b
  
        p   = softmax(self.y)     
        return np.random.choice(range(self.lenOut), p=p.ravel())
  
    def get_prob(self,x):
        # update input
        self.x = x
        self.LSTM_Rec.hx = np.hstack((self.h[:self.hiddenRec], self.x[:self.lenInRec]));
        self.LSTM_Rec.dvo = np.ones(self.hiddenRec)   
        cR, hR, fR, iR, mR, oR = self.LSTM_Rec.fwd_pass()            
        self.c = cR
        self.h[:self.hiddenRec] = hR

        # Feed forward part 
        self.FF.x  = self.x[self.lenInRec:];
        self.FF.dvo = np.ones(self.sizeHidden-self.hiddenRec)
        hN = self.FF.fwd_pass()            
        self.h[self.hiddenRec:] = hN
        
        if(self.extra_fc == 1):
            self.h2 = tanh(np.dot(self.W2,self.h) + self.b2)
            self.y = np.dot(self.W,self.h2) + self.b
        else:
            self.y = np.dot(self.W,self.h) + self.b
            
        self.prob = softmax(self.y)             
        self.is_target = shifted_sigmoid(self.is_target + self.leaky_W * self.prob[1] - self.leaky_b)
        self.is_target = np.clip(self.is_target, 0, 1);       
        
        return self.is_target
    


# In[2]:

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
        
        # for dropout
        self.dvo = np.zeros(sizeHidden); 
        ''' end of LSTM.__init__ '''
        
    def fwd_pass(self):
        f       = sigmoid(np.dot(self.fW, self.hx) + self.fb)
        i       = sigmoid(np.dot(self.iW, self.hx) + self.ib)
        m       = tanh(   np.dot(self.mW, self.hx) + self.mb)        
        o       = sigmoid(np.dot(self.oW, self.hx) + self.ob)
        o       = np.multiply(o,self.dvo); # dropout
        self.c *= f
        self.c += i * m
        self.h  = o * tanh(self.c)
        
        return self.c, self.h, f, i, m, o;
    
    def bwd_pass(self, dh, prev_c, f, i, m, o, c_g):
        
        dh = np.clip(dh, -6, 6);       
        # h = o*tanh(c)
        do  = tanh(self.c) * dh
        do  = sigmoid_deriv(o)*do
        #do  = np.multiply(do,self.dvo)
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
        hx_grad = np.dot(self.fW.T, df) + np.dot(self.iW.T, di) + np.dot(self.oW.T, do) + np.dot(self.mW.T, dm)
        
        
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
        


# In[4]:

class FF:
    
    def __init__ (self,lenIn,sizeHidden,lenRec,learningRate):
        self.lenIn        = lenIn
        self.sizeHidden   = sizeHidden
        self.lenRec       = lenRec
        self.learningRate = learningRate
        
        # hx == x is x and h horizontally stacked together
        self.x = np.zeros(lenIn)
        self.h = np.zeros(sizeHidden)
        
        # Weight matrices
        self.hW = np.random.random((sizeHidden,lenIn));
                            
        # biases
        self.hb = np.zeros(sizeHidden);
             
        # for RMSprop only
        self.GhW = np.random.random((sizeHidden,lenIn));
        self.Ghb = np.zeros(sizeHidden); 
        
        # for dropout
        self.dvo = np.zeros(sizeHidden); 
        ''' end of LSTM.__init__ '''
        
    def fwd_pass(self):
        self.h       = sigmoid(np.dot(self.hW, self.x) + self.hb)
        #self.h       = np.multiply(self.h,self.dvo); # dropout
        return self.h;
    
    def bwd_pass(self, dh):
        
        dh = np.clip(dh, -6, 6);       
        dh  = sigmoid_deriv(self.h)*dh
        dhb = dh;
        dhW = np.dot((np.atleast_2d(dh)).T,np.atleast_2d(self.x)) 
        
        return dhW, dhb;
    
    def update(self, hW_grad, hb_grad):

        self.GhW = 0.9*self.GhW+ 0.1*hW_grad**2
        self.Ghb = 0.9*self.Ghb+ 0.1*hb_grad**2
        
        self.hW -= self.learningRate/np.sqrt(self.GhW + 1e-8) * hW_grad
        self.hb -= self.learningRate/np.sqrt(self.Ghb + 1e-8) * hb_grad
        


# In[ ]:



