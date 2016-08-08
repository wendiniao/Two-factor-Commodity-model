# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:33:57 2016

@author: wendi.zhu
"""

import os
import numpy as np
os.getcwd()
os.chdir("put your working directory here")
os.listdir()
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_slsqp
mat = scipy.io.loadmat('data.mat')
mat_content=mat["ss2000OilData"]

len_maturity=mat_content.shape[1]
predict_f=[]
num_train=8
parametersth=[0.1,0.2,0.08,0,0.1,0.08,0.25]
maturities=np.array([i+1 for i in range(0,len_maturity)]).reshape((1,len_maturity))
dt=1
chi0=0.0
psi0=3
tenor_f=1
#replace the following implied vol given the maturities
implied_vol=np.array([0.39,0.39,0.39,0.39,0.40,0.41	,0.41,0.41,0.41,0.40,0.40,0.40,0.39,0.39,0.38,0.37,	0.37,	0.36,	0.35,	0.35,	0.34,	0.34	,0.33,0.33,	0.32	,0.32	,0.31,0.31	,0.30,0.30,	0.30,	0.29,	0.29,	0.29	,0.28,0.28	,0.28	,0.27	,0.27,0.27	,0.27,0.26,	0.26,0.26,0.26,0.25,0.25,0.25,0.25,0.25	,0.25,0.24,	0.24,	0.24,0.24,	0.24,	0.24	,0.23	,0.23,0.23,	0.23,	0.23,	0.23,	0.23,	0.23,	0.23,	0.23,	0.22,	0.22,0.22,0.22,0.22]).reshape((1,len_maturity))
a0=np.array([chi0,psi0]).reshape((2,tenor_f))  
P0=np.identity(2)*100

#use first 5 observation to train and set the initial optimal value
observation=mat_content[0:5]

def A_T(T,mu_eps_star,mu_eps,k,lambda_chi,ag_chi,ag_eps,rho):
    term1=mu_eps_star*T
    term2=-(1-np.exp(-k*T))*lambda_chi/k
    term3=(1-np.exp(-2*k*T))*ag_chi*ag_chi/(2*k)*ag_eps*ag_eps*T
    term4=2*(1-np.exp(-k*T))*rho*ag_chi*ag_eps/k
    term5=0.5*(term3+term4)
    return term1+term2+term5 
    
def futprice(chi0,psi0,T,mu_eps_star,mu_eps,k,lambda_chi,ag_chi,ag_eps,rho):
    logfut=np.exp(-k*T)*chi0+psi0+A_T(T,mu_eps_star,mu_eps,k,lambda_chi,ag_chi,ag_eps,rho)   
    return np.exp(logfut)
    


def kf_loglik(observation,T,Z,d,H,c,Q,a0,P0):
    loglik=0
    at_upd=[a0.transpose()]
    Pt_upd=P0
    at_pred=[]
    Pt_pred=0
    errory=[]
    for i in range(0,len(observation)):
        at_pred.append((np.dot(T,at_upd[i].transpose())+c).transpose())  
        Pt_pred=np.dot(np.dot(T,Pt_upd),T).transpose()+Q
        yt=np.array(observation[i]).reshape((len_maturity,1))
        F=np.dot(np.dot(Z,Pt_pred),Z.transpose())+H
        IF=np.linalg.inv(F)
        prederror=yt-np.dot(Z,(at_pred[i].transpose()))-d
        errory.append(prederror.transpose())
        Pt_upd=Pt_pred-np.dot(np.dot(np.dot(np.dot(Pt_pred,Z.transpose()),IF),Z),Pt_pred)
        at_upd.append(at_pred[i]+np.dot(np.dot(np.dot(Pt_pred,Z.transpose()),IF),prederror).transpose())
        forward_f=np.dot(Z,(at_pred[-1].transpose()))+d
        loglik=loglik-0.5*np.log10(np.abs(np.linalg.det(F)))-0.5*np.dot(prederror.transpose(),np.dot(IF,prederror))
    return (loglik,forward_f)
    
    
    

def kf_loglik_max(parametersth):
    k=parametersth[0]*parametersth[0]
    ag_chi=parametersth[1]*parametersth[1]
    lambda_chi=parametersth[2]
    mu_eps=parametersth[3]
    ag_eps=parametersth[4]*parametersth[4]
    mu_eps_star=parametersth[5]
    rho_eps_chi=parametersth[6]/(1+abs(parametersth[6]))
    ag2_H=np.array(implied_vol*implied_vol).reshape((1,len_maturity))
    nmat=maturities.shape[1]
    Z=np.concatenate((np.exp(-k*maturities),np.ones((1,nmat))),axis=0).transpose()
    T=np.array([np.exp(-k*dt),0,0,1]).reshape((2,2))
    d=A_T(maturities.transpose(),mu_eps_star,mu_eps,k,lambda_chi,ag_chi,ag_eps,rho_eps_chi)
    H=ag2_H*np.identity(nmat)
    c=np.array([0,mu_eps*dt]).reshape((2,1))
    Q=np.array([(1-np.exp(-2*k*dt))*ag_chi*ag_chi/(2*k),(1-np.exp(-k*dt))*rho_eps_chi*ag_chi*ag_eps/(k),(1-np.exp(-k*dt))*rho_eps_chi*ag_chi*ag_eps/(k), ag_eps*ag_eps*dt]).reshape((2,2))
    loglik=-kf_loglik(observation,T,Z,d,H,c,Q,a0,P0)[0]   
    return loglik

def test_ieqcons(x):
    return np.array((np.ones(len(x))-x)*x )
result = fmin_slsqp(kf_loglik_max, parametersth,iprint=2,f_ieqcons=test_ieqcons,full_output=1)
parametersth=result[0]


def forwardprice(parametersth,observation_f):
    k=parametersth[0]*parametersth[0]
    ag_chi=parametersth[1]*parametersth[1]
    lambda_chi=parametersth[2]
    mu_eps=parametersth[3]
    ag_eps=parametersth[4]*parametersth[4]
    mu_eps_star=parametersth[5]
    rho_eps_chi=parametersth[6]/(1+abs(parametersth[6]))
    ag2_H=np.array(implied_vol*implied_vol).reshape((1,len_maturity))
    nmat=maturities.shape[1]
    Z=np.concatenate((np.exp(-k*maturities),np.ones((1,nmat))),axis=0).transpose()
    T=np.array([np.exp(-k*dt),0,0,1]).reshape((2,2))
    d=A_T(maturities.transpose(),mu_eps_star,mu_eps,k,lambda_chi,ag_chi,ag_eps,rho_eps_chi)
    H=ag2_H*np.identity(nmat)
    c=np.array([0,mu_eps*dt]).reshape((2,1))
    Q=np.array([(1-np.exp(-2*k*dt))*ag_chi*ag_chi/(2*k),(1-np.exp(-k*dt))*rho_eps_chi*ag_chi*ag_eps/(k),(1-np.exp(-k*dt))*rho_eps_chi*ag_chi*ag_eps/(k), ag_eps*ag_eps*dt]).reshape((2,2))
    chi0=observation_f[0][0]/31
    psi0=observation_f[0][0]/31*30
    a0=np.array([chi0,psi0]).reshape((2,tenor_f))  
    P0=np.identity(2)*100  
    price_f=kf_loglik(observation_f,T,Z,d,H,c,Q,a0,P0)[1]
 #   at_upd=kf_loglik(observation,T,Z,d,H,c,Q,a0,P0)[2][-1]
 #   Pt_upd=kf_loglik(observation,T,Z,d,H,c,Q,a0,P0)[3]
    return price_f
 #   return [price_f,at_upd,Pt_upd]
    
for i in range(1,15):
   observed_f=mat_content[5:5+i]
   price_f=forwardprice(parametersth,observed_f)
   observed_f=mat_content[5+i]
   print(observed_f.reshape((72,1)),price_f)
   plt.figure()
   plt.plot(price_f,'r',observed_f,'b')