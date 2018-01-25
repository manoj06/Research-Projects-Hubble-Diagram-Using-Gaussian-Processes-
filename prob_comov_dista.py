import numpy as np
from scipy import integrate as integ
from scipy.integrate import quad
import math as mth
import matplotlib
from sympy import *
import math
from scipy import integrate
import scipy
from scipy.misc import derivative
import matplotlib.pyplot as plt
from gapp import dgp
import pickle
from numpy import array,concatenate,loadtxt,savetxt,zeros
from scipy import integrate
import matplotlib.mlab as mlab
def R_H_ct(z):
	H_0=63.003
	return H_0*(1+z)
def LCDM(z):
	H_0=70.5
	m=0.22*((1+z)**3.0)
	l=0.78
	k_l=(m+l)**(1.0/2)
	return H_0*k_l
def WCDM(z):
	H_0=64.3
	m=0.3*((1+z)**3.0)
	l=0.7
	k_l=(m+l)**(1.0/2)
	return H_0*k_l
def Fried(z):
	H_0=72.0
	m=0.3*((1+z)**3.0)
	k=0.7*((1+z)**2.0)
	k_l=(m+k)**(1.0/2)
	return H_0*k_l
def Conc(z):
	H_0=67.4
	m=0.314*((1+z)**3.0)
	l=0.686
	k_l=(m+l)**(1.0/2)
	return H_0*k_l
def Einst(z):
	H_0=53.5
	m=(1+z)**(3.0/2)
	return H_0*m
#*********************************************************
def R_H_ct_D(z):
	H_0=63.003
	return (H_0*(1+z))**(-1.0)
def LCDM_D(z):
	H_0=70.5
	m=0.28*((1+z)**3.0)
	l=1-0.28
	k_l=(m+l)**(1.0/2)
	return (H_0*k_l)**(-1.0)
def WCDM_D(z):
	H_0=66.0
	m=0.314*((1+z)**3.0)
	l=(1-0.314)*((1+z)**(3*(0.1625)))
	k_l=(m+l)**(1.0/2)
	return (H_0*k_l)**(-1.0)
def Fried_D(z):
	H_0=72.0
	m=0.3*((1+z)**3.0)
	k=0.7*((1+z)**2.0)
	k_l=(m+k)**(1.0/2)
	return (H_0*k_l)**(-1.0)
def Conc_D(z):
	H_0=67.4
	m=0.314*((1+z)**3.0)
	l=0.686
	k_l=(m+l)**(1.0/2)
	return (H_0*k_l)**(-1.0)
def Einst_D(z):
	H_0=41.28
	m=(1+z)**(3.0/2)
	return (H_0*m)**(-1.0)
#*********************************************************
def D_R(z):
	val=integrate.quad(R_H_ct_D,0,z)[0]
	return val
def D_L(z):
	val=integrate.quad(LCDM_D,0,z)[0]
	return val
def D_W(z):
	val=integrate.quad(WCDM_D,0,z)[0]
	return val
def D_F(z):
	val=integrate.quad(Fried_D,0,z)[0]
	return val
def D_E(z):
	val=integrate.quad(Einst_D,0,z)[0]
	return val
def D_C(z):
	val=integrate.quad(Conc_D,0,z)[0]
	return val
if __name__=="__main__":
 	# load data
	(z_z,H_z,Sigma) = loadtxt("h_z.txt",unpack='True')
	(z_data,H_data,sig_data)=loadtxt("Data_obtained.txt",unpack='True')
	# Gaussian process
	"""g = dgp.DGaussianProcess(z_z,H_z,Sigma,cXstar=(0.0,2.1,200),theta=[1,120])
	(rec,theta) = g.gp()
	(drec,theta) = g.dgp(thetatrain='False')
	(d2rec,theta) = g.d2gp()
	savetxt("D6.txt",rec)
	z_data=np.zeros(len(rec))
	H_data=np.zeros(len(rec))
	sig_data=np.zeros(len(rec))
	for i in range(len(rec)):
		z_data[i]=rec[i][0]
		H_data[i]=rec[i][1]
		sig_data[i]=rec[i][2]"""
	D_com_sta=np.zeros(len(z_data)-1)
	X_com_sta=np.zeros(len(z_data)-1)
	D_rh=np.zeros(len(z_data)-1)
	D_lcdm=np.zeros(len(z_data)-1)
	D_Conc=np.zeros(len(z_data)-1)
	D_Eins=np.zeros(len(z_data)-1)
	D_fried=np.zeros(len(z_data)-1)
	D_wcdm=np.zeros(len(z_data)-1)
	for i in range(len(z_data)-1):
		count=0.0
		for j in range(i):
			delta_z=z_data[j+1]-z_data[j]
			H_j_1=1/(H_data[j+1])
			H_j=1/(H_data[j])
			H_jj=H_j_1+H_j
			count=count+(delta_z*H_jj)
		D_com_sta[i]=(count/2)*H_data[0]
		X_com_sta[i]=z_data[i]
		D_rh[i]=D_R(z_data[i])*H_data[0]
		D_lcdm[i]=D_L(z_data[i])*H_data[0]
		D_Conc[i]=D_C(z_data[i])*H_data[0]
		D_Eins[i]=D_E(z_data[i])*H_data[0]
		D_wcdm[i]=D_W(z_data[i])*H_data[0]
		D_fried[i]=D_F(z_data[i])*H_data[0]
	area_sta=integrate.simps(D_com_sta,X_com_sta)
	area_rh=integrate.simps(np.abs(D_rh-D_com_sta),X_com_sta)/area_sta
	area_lcdm=integrate.simps(np.abs(D_lcdm-D_com_sta),X_com_sta)/area_sta
	area_wcdm=integrate.simps(np.abs(D_wcdm-D_com_sta),X_com_sta)/area_sta
	area_Conc=integrate.simps(np.abs(D_Conc-D_com_sta),X_com_sta)/area_sta
	area_Eins=integrate.simps(np.abs(D_Eins-D_com_sta),X_com_sta)/area_sta

	#########################################################################################
	#Cummulative
	area=np.zeros(1000)
	for i in range(1000):
		Y_H_gen=np.zeros(len(H_z))
		for j in range(len(Y_H_gen)):
			r_v=np.random.randn()
			f_g=H_z[j]+r_v*Sigma[j]
			Y_H_gen[j]=f_g
		g_gen=dgp.DGaussianProcess(z_z,Y_H_gen,Sigma,cXstar=(0.0,2.1,200),theta=[1,120])
		(rec_gen,theta_gen)=g_gen.gp()
		Y_gen_data=np.zeros(len(rec_gen))
		for di in range(len(rec_gen)):
			Y_gen_data[di]=rec_gen[di][1]
		D_com_gen=np.zeros(len(z_data)-1)
		for lk in range(len(z_data)-1):
			count=0.0
			for kl in range(lk):
				delta_lk=z_data[kl+1]-z_data[kl]
				H_lk_1=1.0/(Y_gen_data[kl+1])
				H_lk_2=1.0/(Y_gen_data[kl])
				H_lk=H_lk_1+H_lk_2
				count=count+(delta_lk*H_lk)
			D_com_gen[lk]=(count/2.0)*H_data[0]
		area_1=(integrate.simps(np.abs(D_com_gen-D_com_sta),X_com_sta))/area_sta
		area[i]=area_1		
	#########################################################################################
	
	"""#Errors D_comoving
	Sig_D_sta=np.zeros(len(z_data)-1)
	for i in range(len(z_data)-1):
		count=0.0
		for j in range(i):
			delta_z=z_data[j+1]-z_data[j]
			e_1=(sig_data[j+1]**2)/(H_data[j+1]**4)
			e_2=(sig_data[j]**2)/(H_data[j]**4)
			e_3=(e_1+e_2)**(1.0/2.0)
			e_4=e_3*delta_z
			e_5=e_4
			count=count+e_5
		Sig_D_sta[i]=(count/(2.0))*41.28#*(3*10**8)
	#print Sig_D_sta"""
	###########################################################################################
	cum_gen,bins_gen,patches_gen=plt.hist(area,bins=100,normed='True',histtype='step',cumulative=True)
	print "R_h=",area_rh
	print "LCDM=",area_lcdm
	print "WCDM=",area_wcdm
	print "Eins=",area_Eins
	print "Conc",area_Conc
	for i in range(len(bins_gen)-1):
		if((area_rh>bins_gen[i]) and (area_rh<bins_gen[i+1])):
			prob_rh=cum_gen[i]
			point_rh=bins_gen[i]
			print "prob_rh=",cum_gen[i]
			plt.plot(point_rh,prob_rh,"ro")
		if((area_lcdm>bins_gen[i]) and (area_lcdm<bins_gen[i+1])):
			prob_lcdm=cum_gen[i]
			point_lcdm=bins_gen[i]
			print "prob_lcdm=",cum_gen[i]
			plt.plot(point_lcdm,prob_lcdm,"ro")
		if((area_wcdm>bins_gen[i]) and (area_wcdm<bins_gen[i+1])):
			prob_wcdm=cum_gen[i]
			point_wcdm=bins_gen[i]
			print "prob_wcdm=",cum_gen[i]
			plt.plot(point_wcdm,prob_wcdm,"ro")
		if((area_Eins>bins_gen[i]) and (area_Eins<bins_gen[i+1])):
			prob_Eins=cum_gen[i]
			point_Eins=bins_gen[i]
			print "prob_Eins=",cum_gen[i]
			plt.plot(point_Eins,prob_Eins,"ro")
		if((area_Conc>bins_gen[i]) and (area_Conc<bins_gen[i+1])):
			prob_Conc=cum_gen[i]
			point_Conc=bins_gen[i]
			print "Prob_Conc=",cum_gen[i]
			plt.plot(point_Conc,prob_Conc,"ro")
	
	plt.show()	
	#print "Fried",area_Fried
	#plt.plot(X_com_sta,D_com_sta,color='black')
	#plt.plot(X_com_sta,D_com_sta+Sig_D_sta,color='red')
	#plt.plot(X_com_sta,D_com_sta-Sig_D_sta,color='red')
	#plt.plot(X_com_sta,D_com_sta+2*Sig_D_sta,color='pink')
	#plt.plot(X_com_sta,D_com_sta-2*Sig_D_sta,color='pink')
	#plt.plot(X_com_sta,D_Eins,color='blue')
	#plt.axis([0,2,0,1.25])
	#plt.title("$D_{comov}$ and $D_{Eins}$")
	#plt.show()
