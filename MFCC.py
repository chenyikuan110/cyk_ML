def get_mfcc_filter_bank(Sxx_in,num_bank,fs):
    m = np.linspace(start=1125 * np.log(1 + 10 / 700.0),stop=1125 * np.log(1 + fs*0.8 / 700.0),num=num_bank)
    # get list of freq centers
    print(m)
    f = np.zeros(len(m)+1)
    f[0:len(m)] = 700 * (np.exp(m / 1125) - 1)
    # get list of indices
    f = np.floor((Sxx_in.shape[1]+1)*f/fs)
    f[len(m)] = int(f[num_bank-1]*f[num_bank-1]/f[num_bank-2])
    print(f)
    # get coef for each filter bank for all time steps
    filter_bank = np.zeros((num_bank,Sxx_in.shape[1]))
    for i in range(0,num_bank):             
        for k in range(Sxx_in.shape[1]):           
            if(f[i-1] <= k and k <= f[i]):
                filter_bank[i][k] = (k-f[i-1]) / (f[i] - f[i-1]);#/(f[i+1] - f[i-1])*2
            elif(f[i] <= k and k <= f[i+1]):
                filter_bank[i][k] = (f[i+1]-k) / (f[i+1] - f[i]);#/(f[i+1] - f[i-1])*2
                
    return filter_bank;

def Sxxin_to_mfcc(Sxx_in,num_bank,fs,filter_bank):
    from scipy import fftpack
    mfcc_mat = np.zeros((Sxx_in.shape[0],num_bank)) # dim = (t,num_bank), same as Sxx_in

    for t in range(0,Sxx_in.shape[0]):
        mfcc_mat[t] = fftpack.dct(np.log(np.dot(filter_bank,Sxx_in[t])))
    
    print(mfcc_mat.shape)
    return mfcc_mat;

def mfcc_difference(mfcc_mat):
    diff_mat = np.zeros_like(mfcc_mat);
    
    for t in range(1,mfcc_mat.shape[0]):
        diff_mat[t] = mfcc_mat[t] - mfcc_mat[t-1]
    return diff_mat
	
# mfcc test
num_bank = 10
Sxx_in,labels,fs = audio_prep("./audio_set/data/file_00001.wav","./audio_set/text/file_00.txt",plot=0,listen=0,noise=0)
filter_bank = get_mfcc_filter_bank(Sxx_in[0:1],num_bank,fs)
for i in range(num_bank):
    plt.plot(filter_bank[i])
plt.show()
print("Test mfcc:")
Sxxin_to_mfcc(Sxx_in,num_bank,fs,filter_bank)
      

epoch = 30*1000;
Sxx_spec,labels,fs = audio_prep("./audio_set/data/file_00001.wav","./audio_set/text/file_00.txt",plot=1,listen=1,noise=0)

num_bank = 10
filter_bank = get_mfcc_filter_bank(Sxx_spec[0:1],num_bank,fs)
Sxx_mfcc = Sxxin_to_mfcc(Sxx_spec,num_bank,fs,filter_bank)
Sxx_mfcc_diff = mfcc_difference(Sxx_mfcc)

Sxx_in = np.hstack((Sxx_spec,Sxx_mfcc,Sxx_mfcc_diff))

lenIn, lenOut, lenRec = Sxx_in.shape[1],2, min(150,max(min(100,Sxx_in.shape[0]),int(Sxx_in.shape[0]/10)))
sizeHidden, numHiddenLayer = 100,1;
learningRate = 0.1;

print("lenIn=",lenIn, " lenOut=",lenOut," lenRec=",lenRec)

# training
from os import listdir
from os.path import isfile, join
txtpath = "./audio_set/text/"
datpath = "./audio_set/data/"
txtfiles = [f for f in listdir(txtpath) if isfile(join(txtpath, f))]
print(txtfiles)
datfiles = [f for f in listdir(datpath) if isfile(join(datpath, f))]
print(datfiles)

# single LSTM
R = myRNN(lenIn, lenOut, lenRec, sizeHidden, Sxx_in, labels, learningRate); biDir = 0;lstm=1 

# bidir LSTM
#R = lstmRNN(lenIn, lenOut, lenRec, sizeHidden, Sxx_in, labels, learningRate); biDir = 1;lstm=1 

# single RNN
#R = basicRNN(lenIn, lenOut, lenRec, sizeHidden, Sxx_in, labels, learningRate); biDir = 0;lstm = 0 

# bidir RNN
#R = basicRNN(lenIn, lenOut, lenRec, sizeHidden, Sxx_in, labels, learningRate); biDir = 1;lstm = 0 

n,k,position,time = 0,0,0,0;
while n<epoch:
    
    if(position+lenRec+1 >= Sxx_in.shape[0] or n == 0):
        if(time > 50):
            if(k == len(txtfiles)):
                k = 0;
            n = 0
            Sxx_spec,labels,fs = audio_prep("./audio_set/data/"+datfiles[k],"./audio_set/text/"+txtfiles[k],plot=0,listen=0,noise=0)
            Sxx_mfcc = Sxxin_to_mfcc(Sxx_spec,num_bank,fs,filter_bank)
            Sxx_mfcc_diff = mfcc_difference(Sxx_mfcc)
            Sxx_in = np.hstack((Sxx_spec,Sxx_mfcc,Sxx_mfcc_diff))
            
            
            print("k updates! k = ",k,"New Sxx has length ",Sxx_in.shape[0])
            print("./audio_set/data/"+datfiles[k])
            k = k+1
            time = 0
        if(biDir == 0):
            R.h = np.zeros_like(R.h)
            if(lstm != 0):
                R.c = np.zeros_like(R.c)
        else:
            R.hls_infer = np.zeros_like(R.hls_infer)
            R.hrs_infer = np.zeros_like(R.hrs_infer)
            if(lstm != 0):
                R.cls_infer = np.zeros_like(R.cls_infer)
                R.crs_infer = np.zeros_like(R.crs_infer)
        position = 0;
        time += 1;
        
    inputs  = Sxx_in[position:position+lenRec]
    targets  = labels[position:position+lenRec]

    R.update_inputs_targets(inputs,targets)
    R.fwd_pass();
    
    err = R.bwd_pass();
    
    if(n%113 == 0 ): # 113 is a prime number
        print("times:",time,"n=",n,"err:",err)
        
        result = []
        compare = []
        ps = 0 
        if(biDir == 0):
            # single layer
            for i in range(0,Sxx_in.shape[0]):
                infer_in_enc = Sxx_in[ps+i]

                ret = R.get_prob(infer_in_enc)
                result.append(ret)
                compare.append(labels[ps+i])
        else:
            # bi-directional
            while(ps+lenRec<Sxx_in.shape[0]):
                infer_in_enc  = Sxx_in[ps:ps+lenRec,:];
                ret = R.inference(infer_in_enc)     
                result.append(ret[:,1])
                compare.append(list(labels[ps:ps+lenRec])[:])
                ps += lenRec
            compare = np.array(compare)
            compare = np.reshape(compare,(compare.shape[0]*compare.shape[1],1))
            result = np.array(result)
            result = np.reshape(result,(result.shape[0]*result.shape[1],1))
        plt.plot(compare)
        plt.plot(result)
        
        plt.show()
    position += lenRec;
    n += 1;
	  