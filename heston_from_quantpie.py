#Import libraries

import numpy as np
import math as math
import cmath as cmath
import matplotlib.pyplot as plt
import time


#Class to hold the relevant functions

class Heston(object):
    
    def __init__(self,S0,K,tau,r,kappa,theta,v0,lamda,sigma,rho): #Constructor for initiating the class
        
        self.x0=math.log(S0);
        self.ln_k=math.log(K);
        self.r=r;
        self.v0=v0;
        self.kappa=kappa;
        self.theta=theta;
        self.lamda=lamda;
        self.sigma=sigma;
        self.rho=rho;
        self.tau=tau;
   
        self.a=kappa*theta;
        self.u=[0.5,-0.5];
        self.b=[kappa+lamda-rho*sigma,kappa+lamda];
        
    def reset_parameters(self,S0,K,tau,r,kappa,theta,v0,lamda,sigma,rho): # Function for resetting the constant parameters
        self.x0=math.log(S0);
        self.ln_k=math.log(K);
        self.r=r;
        self.v0=v0;
        self.kappa=kappa;
        self.theta=theta;
        self.lamda=lamda;
        self.sigma=sigma;
        self.rho=rho;
        self.tau=tau;
   
        self.a=kappa*theta;
        self.u=[0.5,-0.5];
        self.b=[kappa+lamda-rho*sigma,kappa+lamda];       
       
    
    def characteristic_func(self,phi):#Return the characteristic functions f1 and f2, each of which has a real and a complex part
        
        d=[0.0,0.0];g=[0.0,0.0];C=[0.0,0.0];D=[0.0,0.0];edt=[0.0,0.0];gedt=[0.0,0.0]; f=[0.0,0.0];

        for j in range(2):

            temp=self.b[j]-1j*self.rho*self.sigma*phi;

            d[j]=cmath.sqrt(temp**2-self.sigma**2*(2.0*self.u[j]*phi*1j-phi**2));

            g[j]=(temp+d[j])/(temp-d[j]);

            edt[j]=cmath.exp(d[j]*self.tau);
            gedt[j]=1.0-g[j]*edt[j];

            D[j]=(temp+d[j])*(1.0-edt[j])/gedt[j]/self.sigma/self.sigma;
            C[j]=self.r*phi*self.tau*1j+self.a/self.sigma/self.sigma*((temp+d[j])*self.tau-2.0*cmath.log(gedt[j]/(1.0-g[j])));
            f[j]=cmath.exp(C[j]+D[j]*self.v0+1j*phi*self.x0);     
            
        return f;    
 
    def f1(self,phi):#f1 only using a copy of the previous code with minimal change, i.e.,j=0 replaes loop
        
        d=[0.0,0.0];g=[0.0,0.0];C=[0.0,0.0];D=[0.0,0.0];edt=[0.0,0.0];gedt=[0.0,0.0]; f=[0.0,0.0];

        j=0;

        temp=self.b[j]-1j*self.rho*self.sigma*phi;

        d[j]=cmath.sqrt(temp**2-self.sigma**2*(2.0*self.u[j]*phi*1j-phi**2));
        g[j]=(temp+d[j])/(temp-d[j]);

        edt[j]=cmath.exp(d[j]*self.tau);
        gedt[j]=1.0-g[j]*edt[j];

        D[j]=(temp+d[j])*(1.0-edt[j])/gedt[j]/self.sigma/self.sigma;
        C[j]=self.r*phi*self.tau*1j+self.a/self.sigma/self.sigma*((temp+d[j])*self.tau-2.0*cmath.log(gedt[j]/(1.0-g[j])));
        f[j]=cmath.exp(C[j]+D[j]*self.v0+1j*phi*self.x0);

        return f[0];    
    
    def f2(self,phi):# f2 only using a copy of the previous code with minimal change, i.e.,now j=1 replaes loop
        
        d=[0.0,0.0];g=[0.0,0.0];C=[0.0,0.0];D=[0.0,0.0];edt=[0.0,0.0];gedt=[0.0,0.0]; f=[0.0,0.0];

        j=1;

        temp=self.b[j]-1j*self.rho*self.sigma*phi;

        d[j]=cmath.sqrt(temp**2-self.sigma**2*(2.0*self.u[j]*phi*1j-phi**2));
        g[j]=(temp+d[j])/(temp-d[j]);

        edt[j]=cmath.exp(d[j]*self.tau);
        gedt[j]=1.0-g[j]*edt[j];

        D[j]=(temp+d[j])*(1.0-edt[j])/gedt[j]/self.sigma/self.sigma;
        C[j]=self.r*phi*self.tau*1j+self.a/self.sigma/self.sigma*((temp+d[j])*self.tau-2.0*cmath.log(gedt[j]/(1.0-g[j])));
        f[j]=cmath.exp(C[j]+D[j]*self.v0+1j*phi*self.x0);
        
        return f[1];     
      
    def P1_integrand(self,phi): #Returns the integrand  that appears in the P1 formula
        temp=cmath.exp(-1j*phi*self.ln_k)*self.f1(phi)/1j/phi;
        return temp.real;

    def P2_integrand(self,phi):  #Returns the integrand  that appears in the P1 formula
        temp=cmath.exp(-1j*phi*self.ln_k)*self.f2(phi)/1j/phi;
        return temp.real;
    
    def Probabilities(self,a,b,n):  #Compute the two probabilities: a and b are the integration limits, n is the number of intervals
                                    # usually the interval >0 to 100 captures the range that matters, so no need to go to b=infinity!
        pi_i=1.0/math.pi;
        P1=0.5+pi_i*trapzd(self.P1_integrand,a,b,n); #trapzd function is de
        P2=0.5+pi_i*trapzd(self.P2_integrand,a,b,n);
        P=[P1,P2];
        return P;
    
    def price(self,a,b,n):
        Ps=self.Probabilities(a,b,n);
        
        call_price=math.exp(self.x0)*Ps[0]-math.exp(self.ln_k-self.r*self.tau)*Ps[1];
        put_price=call_price-(math.exp(self.x0)-math.exp(self.ln_k-self.r*self.tau));
        
        output={
            "Call price":call_price,
            "Put price":put_price,
            "P1":Ps[0],
            "P2":Ps[1]
        }
        return output;
    
    def plot_f1f2(self): # Plot real parts of the characteristic functions (f1 and f2), 
                         # and the integrands that appear in P1 and P2
        
        n=2000;
        lwr=-50.111;
        upr=50.0311;
        
        x=np.linspace(lwr,upr,n+1);
        fs=[self.characteristic_func(x[i]) for i in range(n+1)];
        
        y1=[fs[i][0].real for i in range(n+1)];
        y2=[fs[i][0].imag for i in range(n+1)];
        y3=[self.P1_integrand(x[i]) for i in range(n+1)];
        y4=[fs[i][1].real for i in range(n+1)]; 
        y5=[fs[i][1].imag for i in range(n+1)];
        y6=[self.P2_integrand(x[i]) for i in range(n+1)];    
        
        fig=plt.figure();
        f1_real=fig.add_subplot(231);
        f1_real.set_title('Real part of F1');
        f1_real.plot(x,y1);
        
        f1_imag=fig.add_subplot(232);
        f1_imag.set_title('Imaginary part of F1');
        f1_imag.plot(x,y2);        
        
        f1_integrand=fig.add_subplot(233);
        f1_integrand.set_title('Integrand of P1');
        f1_integrand.plot(x,y3);      
   
        f2_real=fig.add_subplot(234);
        f2_real.set_title('Real part of F2');
        f2_real.plot(x,y4);
        
        f2_imag=fig.add_subplot(235);
        f2_imag.set_title('Imaginary part of F2');
        f2_imag.plot(x,y5);
        
        f2_integrand=fig.add_subplot(236);
        f2_integrand.set_title('Integrand of P2');
        f2_integrand.plot(x,y6); 
        
        plt.show();
  
  #end class
  
        
def trapzd(func,a,b,n): #Trapzoid method for numerical integration, one can also use a function from scipy.integrate library
        
    if (n<1):
        return ;
    elif (n==1):
        return 0.5*(b-a)*(func(a)+func(b));
    else:
            
        temp=0.0;
        dx=(b-a)/n;
            
        x=np.linspace(a,b,n+1);
        y=[func(x[i]) for i in range(n+1)];
            
        temp=0.5*dx*np.sum(y[1:]+ y[:-1]);
        return temp;
        
        
#Usage Example
hc=Heston(S0=100,K=100*1.1,tau=2,r=0.0157,kappa=1.538,theta=0.0127,v0=0.071,lamda=0.5,sigma=0.0708,rho=-0.796);

price=hc.price(0.00001,100,10000);

print("The prices of European options and the underlying P1 and P2 are presented below:\n")
print(price)

# print("\n\nThe price is based on the following parameters:\n")
# print(vars(hc))
# hc.plot_f1f2()

# S0 = np.linspace(80,100,10)
# for s in S0:
#     t = time.time()
    
#     hc=Heston(S0=s,K=100,tau=1.0,r=0.05,kappa=5.0,theta=0.1,v0=0.1,lamda=0.5,sigma=0.5,rho=-0.7);

#     price=hc.price(0.00001,100,10000);
#     t = time.time() - t

#     print(t,price['Call price'])

# import sys
# print(sys.version)
