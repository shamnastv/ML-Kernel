import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix as cv_mat
from cvxopt import solvers as cvxopt_solvers

def abline(slope, intercept,label="hai"):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',label=label)
def svm(C,train_data,train_label,test_data,test_label,error=1e-6,
        view_progress=False,view_graph=True,box_plot=True):
    p1 = np.array([[0 for i in range (103)] for i in range (103)])
    p1[0][0] = 1
    p1[1][1] = 1
    tc_string = "d"
    P = cv_mat(p1,tc=tc_string)
    q1 = [0,0,0]
    for i in range (100):
        q1.append(C)
    q1 = np.array(q1)
    q = cv_mat(q1,tc=tc_string)
    g11x1=[]
    g11x2=[]
    for i in range(100):
        g11x1.append(-train_data[0,i]*train_label[i])
        g11x2.append(-train_data[1,i]*train_label[i])
    g11x1=np.array(g11x1).reshape(100,1)
    g11x2=np.array(g11x2).reshape(100,1)
    g11 = np.concatenate((g11x1,g11x2),axis =1)
    g12 = -train_label.reshape(100,1)
    g12 = np.concatenate((g11,g12),axis = 1)
    g13 = - np.identity(100)
    g17 = np.concatenate((g12,g13),axis =1)
    g14 = np.zeros((100,3))
    g15 = - np.identity(100)
    g16 = np.concatenate((g14,g15),axis =1)
    g1 = np.concatenate((g17,g16),axis =0)
    G = cv_mat(g1,tc=tc_string)
    h1 = [-1 for i in range(100)]
    h2 = [0 for i in range(100)]
    h1.extend(h2)
    h1 = np.array(h1)
    h = cv_mat(h1,tc=tc_string)
    cvxopt_solvers.options['show_progress'] = view_progress
    cvxopt_solvers.options['abstol'] = error
    cvxopt_solvers.options['reltol'] = error
    cvxopt_solvers.options['feastol'] = error
    cvxopt_solvers.options['maxiters'] = 500

    sol = cvxopt_solvers.qp(P,q,G,h)
    # print(sol["x"])
    x=sol["x"]
    # print(x)
    w=np.array([x[0],x[1]]).reshape(2,1)
    w1,w2= w
    b=x[2]
    predicted_label=[]
    for i in range (100):
        y = np.dot(np.transpose(w),train_data[:,i])+b
        x1,x2 = train_data[:,i]
#         print(y)
#         print(w1*x1+w2*x2+b)
        if(y>0):
            predicted_label.append(+1)
        else:
            predicted_label.append(-1)
    correct_labels=0
    for i in range (100):
        if (train_label[i]==predicted_label[i]):
            correct_labels+=1
    print("C,correct labels out of 100,primal objective,status=",C,correct_labels,sol["primal objective"],sol["status"])
    if(view_graph):
        if(box_plot):
            axes = plt.gca()
            axes.set_xlim([-3.5,3.5])
            axes.set_ylim([-3.5,3.5])
        slope=-w1/w2
        c1= 1-b
        c1= c1/w2
        plt.title("SVM(Only train data)")
        plt.scatter(train_data[0],train_data[1],c=train_label)
        abline(slope[0],c1[0],label="wx+b=1")
        c2= -b/w2
        abline(slope[0],c2[0],label='wx+b=0')
        c3= (-1-b)/w2
        abline(slope[0],c3[0],label='wx+b=-1')
        plt.legend()
        plt.grid()
        plt.plot()   
    return(w1[0],w2[0],b)
def test(test_data,test_label,w1,w2,b):
    predicted_label=[]
    for i in range (50):
        x1,x2 = test_data[:,i]
        y=w1*x1+w2*x2+b
        if(y>0):
            predicted_label.append(+1)
        else:
            predicted_label.append(-1)
    correct_labels=0
    for i in range (50):
        if (test_label[i]==predicted_label[i]):
            correct_labels+=1
    print("correct out of 50=",correct_labels)
def gen_points_2a():
#     generator for points in 2a _ both train and test ,ie 150
    np.random.seed(600)
    w_sample = np.random.multivariate_normal(mean=[0,0],cov=[[1,0],[0,1]]).reshape(2,1)
    b_sample = np.random.normal(0,1)
    x0_samples = np.random.uniform(-3,3,150).reshape(1,150)
    x1_samples = np.random.uniform(-3,3,150).reshape(1,150)
    x_samples =np.concatenate((x0_samples,x1_samples),axis=0)
    classes = []
    for i in range (150):
        y = np.dot(np.transpose(w_sample),x_samples[:,i])+b_sample
        if(y>0):
            classes.append(1)
        else:
            classes.append(-1)
    classes= np.array(classes)
    train_data = x_samples[:,:100]
    train_label = classes[:100]
    test_data = x_samples[:,100:]
    test_label = classes[100:]
    return(train_data,train_label,test_data,test_label)
    
train_data,train_label,test_data,test_label=gen_points_2a()
plt.figure(1)
plt.title("Distribution of train_data")
plt.scatter(train_data[0],train_data[1],c=train_label)
plt.plot()
plt.show()
# plt.savefig("./Plot/2a.jpg")
plt.figure(2)
##Question B
C=50
w1,w2,b=svm(C,train_data,train_label,test_data,test_label,view_progress =False,view_graph= True)
# plt.savefig("./Plot/2b.jpg")
plt.show()
plt.figure(3)
print("w1,w2,b=",w1,w2,b)
test(test_data,test_label,w1,w2,b)

def gen_points_2c():
#     generator for points in 2a _ both train and test ,ie 150
    np.random.seed(600)
    x0_samples = np.random.uniform(-3,3,150).reshape(1,150)
    x1_samples = np.random.uniform(-3,3,150).reshape(1,150)
    x_samples =np.concatenate((x0_samples,x1_samples),axis=0)
    classes = []
    for i in range (150):
        x1,x2 = x_samples[:,i]
        y1=x1**2+(x2**2)/2
        if(y1<=2):
            classes.append(1)
        else:
            classes.append(-1)         
    classes= np.array(classes)
    train_data = x_samples[:,:100]
    train_label = classes[:100]
    test_data = x_samples[:,100:]
    test_label = classes[100:]
    return(train_data,train_label,test_data,test_label)
    
train_data,train_label,test_data,test_label=gen_points_2c()
plt.title("Distribution of train data")
plt.scatter(train_data[0],train_data[1],c=train_label)
# plt.savefig("./Plot/2c.jpg")
plt.show()
plt.figure(4)


C=1e-7
w1,w2,b=svm(C,train_data,train_label,test_data,test_label,view_progress =False,view_graph= False,
           box_plot=False)
print("w1,w2,b=",w1,w2,b)
test(test_data,test_label,w1,w2,b)

#------------------Question d-----------------------
print(test_data[:,:5])
train_data_d= np.square(train_data)
test_data_d= np.square(test_data)
test_label_d = test_label
train_label_d = train_label
print(test_data_d[:,:5])
plt.title("Distribution of training data")
plt.scatter(train_data_d[0],train_data_d[1],c=train_label_d)
# plt.savefig("./Plot/2d1.jpg")
plt.show()
plt.figure(5)


C=0.8
w1,w2,b=svm(C,train_data_d,train_label_d,test_data_d,test_label_d,view_progress =False,
            view_graph= True,box_plot=False)
axes = plt.gca()
axes.set_xlim([-.5,9.5])
axes.set_ylim([-.5,9.5])
plt.show()
# plt.savefig("./Plot/2d2.jpg")
print("w1,w2,b=",w1,w2,b)
test(test_data_d,test_label_d,w1,w2,b)

#-------------------------------Question e-------------------------------------------
#Reference - https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
def linear_kernal(x,y):
    return(np.dot(x,y))
def gaussian_kernal(x,y,sigma = 1):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
def SVM_dual(C,train_data,train_label,test_data,test_label,sigma=1):
    train_label_1= train_label.reshape(100,1)    
#     X_prime= train_label_1*train_data.T
    K=np.zeros((100,100))
    for i in range (100):
        for j in range (100):
            K[i][j]=gaussian_kernal(train_data.T[i],train_data.T[j],sigma)
    H= np.outer(train_label_1,train_label_1)*K
    tc_string="d"
    P= cv_mat(H,tc=tc_string)
    q= cv_mat(np.array([[-1] for i in range (100)]),tc=tc_string)
    G11 =- np.identity(100)
    G12 =np.identity(100)
    G1= np.concatenate((G11,G12),axis=0)
    G = cv_mat(G1)
    h11 = np.zeros(100)
    h12 = np.zeros(100)+1
    h1 = np.concatenate((h11,h12)).reshape(200,1)
    h= cv_mat(h1,tc=tc_string)
    A= cv_mat(train_label.reshape(1,100),tc=tc_string)
    b= cv_mat(np.array([0]),tc =tc_string)
    cvxopt_solvers.options['show_progress'] = True
    sol=cvxopt_solvers.qp(P,q,G,h,A,b)
    alphas = np.array(sol['x'])
    w= np.dot((train_label_1*alphas).T,train_data.T).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    b = np.mean(train_label_1[S] - np.dot(train_data.T[S], w))
    a = np.ravel(sol['x'])
    # Support vectors have non zero lagrange multipliers
    sv1 = a > 1e-5
    ind = np.arange(len(a))[sv1]
    a = a[sv1]
    sv = train_data.T[sv1]
    sv_y = train_label_1[sv1]
    y_predict = np.zeros(100)
    for i in range(100):
        s = 0
        for al, sv_yl, svl in zip(a,sv_y,sv):
            s += al * sv_yl * gaussian_kernal(train_data.T[i], svl,sigma)
        y_predict[i] = s
    y_predict=y_predict + b
    correct_label= 0
    classes=[]
    for i in range (100):
        if(y_predict[i]>0):
            classes.append(1)
        else:
            classes.append(-1)
    for i in range (100):
        if (classes[i]==train_label[i]):
            correct_label+=1
    print("correct out of 100",correct_label) 
    return(b,a,sv_y,sv)
    

# train_data.shape
# print(train_data[:,:5])
# print(train_label[:5])
# print(train_label*train_data)
sigma =4e-1
C=0.8
y_predict = np.zeros(100)
b,a,sv_y,sv=SVM_dual(C,train_data,train_label,test_data,test_label,sigma)
for i in range(50):
    s = 0
    for al, sv_yl, svl in zip(a,sv_y,sv):
        s += al * sv_yl * gaussian_kernal(test_data.T[i], svl,sigma)
    y_predict[i] = s
y_predict=y_predict + b
correct_label= 0
classes=[]
for i in range (50):
    if(y_predict[i]>0):
        classes.append(1)
    else:
        classes.append(-1)
for i in range (50):
    if (classes[i]==test_label[i]):
        correct_label+=1
print("correct out of 50",correct_label) 


