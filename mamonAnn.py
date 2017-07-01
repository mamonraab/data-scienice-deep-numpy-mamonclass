import numpy as np

class mamonAnn():

    def __init__(self):
        self.W = []
        self.b = []
        self.hedins = []
    def Layers(self , inpt_size , hdeins , output_size):
        # wights init
        for i in xrange(len(hdeins)):
            if i == 0:
                self.W.append(np.random.randn(inpt_size, hdeins[i]) / np.sqrt(inpt_size + hdeins[i]))
                self.b.append(np.zeros(hdeins[i]))
            #elif i == len(hdeins)-1:
            else:
                self.W.append(np.random.randn(hdeins[i-1], hdeins[i]) / np.sqrt(hdeins[i-1] + hdeins[i]))
                self.b.append(np.zeros(hdeins[i]))
            self.hedins.append(np.zeros(hdeins[i]))
        self.b.append(np.zeros(output_size))
        self.W.append(np.random.randn(hdeins[-1], output_size) / np.sqrt(hdeins[-1] + output_size))

    def farowrds(self , X,factive,inneractive,lactive):

        for i in xrange(len(self.hedins)):

            if i == 0:
                self.hedins[i]=calulate_layers(X , self.W[i] , self.b[i] , factive)
            else:
                self.hedins[i]=calulate_layers(self.hedins[i-1] , self.W[i] , self.b[i] , inneractive)
        self.out=calulate_layers(self.hedins[-1] , self.W[-1] , self.b[-1] , lactive)
    def back_train(self ,X , T , lr , reg):
        regularization = reg

        #S_y = T - self.out #S_final(self.out , T)
        T=T.reshape(self.out.shape)
        S_y = T - self.out
        xc = self.hedins[-1].T.dot(S_y)
        yc = regularization * self.W[-1]
        dc = xc - yc


        self.W[-1] += lr * dc
        self.b[-1] -= lr*(S_y.sum() - regularization * self.b[-1])
        S_yold = S_y
        W2 = self.W[-1]
        # gradient descent
        #W2 -= lr*self.hedins[-1].T.dot(self.out - T)
        #b2 -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
        #dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
        #W1 -= learning_rate*Xtrain.T.dot(dZ)
        #b1 -= learning_rate*dZ.sum(axis=0)
        for i in xrange(len(self.hedins)-1,-1,-1):

            if i == 0:

                in_layer = X
            else:
                in_layer = self.hedins[i-1]
            S_flayer = S_anylayer(self.hedins[i] , W2 , S_y)
            self.W[i] += lr *(in_layer.T.dot(S_flayer)  - regularization * self.W[i]  )#delta_Fast(lr ,T , self.out , in_layer, 'any' ,  self.hedins[i] , W2)
            self.b[i] += lr*(S_flayer.sum(axis=0)  - regularization * self.b[i]  )
            W2 = self.W[i]
            S_y = S_flayer
    def fit(self,X,T,forlong , prnt,lr,batch ,reg , factive , mid , last , LL):
        last_error_rate = None
        for trn in xrange(forlong):
        #I , j = X.shape
        #for i in xrange(I):
            self.farowrds(X,factive , mid , last)
            ll = cost(T, self.out)
            LL.append(ll)
            #ll = cross_entropy(T, self.out)
            er = np.mean(self.out != T)
            if er != last_error_rate:
                last_error_rate = er
            Y_old = self.out
            #if i % batch == 0:
            self.back_train(X , T,lr ,reg)
            #if np.array_equal(self.out,Y_old):
                #print("cant improve more itration num is ",trn)
                #return self.out
            if trn % prnt == 0:
                print ll
                #print "  itration  ",trn,'target is ',T ,'predcation is ',self.out

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))


#backpropgation

def onehot_target(y,labels):
    n = len(y)
    T = np.zeros((n,labels))
    for i in xrange(n):
        T[i,y[i]] = 1


def New_weight_Fast(old_w , delta_w):
    return old_w + delta_w
def delta_Fast(lr ,T , Y , inpt , layer , Z , W2):
    if layer == 'last':
        return layer.dot(inpt)
    else:
        S = lr
        dZ = lr * Z * (1 - Z)
        ret2 = inpt.T.dot(dZ)
        return ret2
#retrun array of final S
def S_final(Y , Target):
    S = law(Y , Target)
    return S

#return array of S for any layer except the final
def S_anylayer(layer , layer_w , next_s):
    S = np.zeros(layer.shape)
    K , _  = layer.shape
    if len(layer_w.shape) == 1:
        for k in xrange(K):
            for i in xrange(layer_w.shape):
                dos = layer_w[i] * next_s[k,0]
                coz = dos + S[k,i]
                S[k,i] = coz
                joso = S[k,i] * F_hate(layer[k,i])
                S[k,i] = joso
    else:
        I , J = layer_w.shape
        for k in xrange(K):
            for i in xrange(I):
                for j in xrange(J):
                    dos = layer_w[i,j] *   next_s[k,j]
                    coz = dos + S[k,i]
                    S[k,i] = coz
                joso = S[k,i] * F_hate(layer[k,i])
                S[k,i] = joso
    return S

def law(y , t):
    return t - y
def F_hate(y):
    return y * (1-y)
def weight_up(lr , W , inpt , S):
    if len(W.shape) == 1:
        j = 0
        I = W.shape
        for i in xrange(I[0]):
            W[i] = New_weight(W[i] , delta(lr , i , j , inpt , S))
    else:
        I , J = W.shape
        for i in xrange(I):
            for j in xrange(J):
                W[i,j] = New_weight(W[i,j] , delta(lr , i , j , inpt , S))
    return W

def cost(T, Y):
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y))




def y2indicator(y):
    y = y.astype(np.int32)
    K = len(set(y))
    N = len(y)
    ind = np.zeros((N, K))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind



def activation(act_type , a):
    if act_type == 'softmax':
        if len(a.shape) ==1:
            result =  (np.exp(a)) / (np.exp(a).sum())
        else:
            result = (np.exp(a)) / ( np.exp(a).sum(axis=1 ,  keepdims = True) )
    elif act_type == 'tanh':
        result = np.tanh(a)
    elif act_type == 'relu':
        result = a * (a > 0)
    else:
        #these is sigmoid
        result = 1 / (1 + np.exp(-a))
    return result

def calulate_layers(inpt , wights , byais , act_type):
    return activation(act_type,a=inpt.dot(wights) + byais)


def normalize(data):
    return (data - data.mean()) / data.std()

def catgoricl(data , place):
    # doing one hote encoder for day time
    from sklearn.preprocessing import OneHotEncoder
    #this OneHotEncoder  the param categorical_features mean the colmn that need to be  processed
    onehotencoder = OneHotEncoder(categorical_features = [place])
    return onehotencoder.fit_transform(data).toarray()


def catg2(data , place):
    size = len(set(data[:,place]))
    #size = size - 1
    N , D = data.shape
    z = np.zeros((N ,size ))
    #print z.shape
    z[np.arange(N) ,data[:,place].astype(np.int32) ] = 1
    x2 = np.zeros((N,D+size-1)).astype(np.float32)
    x2[:,0:(place)] = data[:,0:(place)]
    x2[:,-size:] = z
    return x2
