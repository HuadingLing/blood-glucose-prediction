import matplotlib.pyplot as plt
import numpy as np

X=np.linspace(-np.pi,np.pi,256,endpoint=True)#-π to+π的256个值
C,S=np.cos(X),np.sin(X)
plt.plot(X,np.abs(C),'g')
plt.plot(X,S,'b')
plt.xlim(-4,4)
plt.ylim(-2,2)
plt.xlabel('time')
plt.ylabel('glucose level')
plt.title('Prediction result')
plt.show()