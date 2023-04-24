import numpy as np

class odeSolver(object):
	def __init__(self,**kwargs):
		#Initials
		self.simVars=[]
		self.maxStepSize=0.1 #max step size
		self.minStepSize=1e-8 #min step size
		self.errorThresh=1e-5 #Error treshold
		allowed_keys = ['iniStepSize', 'maxStepSize','minStepSize','errorThresh','fun','tSpan','iniCond','simVars']

		self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
		self.resY=self.iniCond

	def solveSystem(self):

		h=self.iniStepSize
		
		t=[]
		iii=0
		t.append(self.tSpan[0])
		y=np.zeros((1,self.iniCond.size))

		y[0,:]=self.iniCond

		while t[iii]< self.tSpan[1]:
			k1=self.fun(t[iii],y[iii,:],self.simVars)
			k2=self.fun(t[iii]+h/4,y[iii,:]+h/4*k1,self.simVars)
			k3=self.fun(t[iii]+3*h/8,y[iii,:]+3*h/32*k1+9/32*h*k2,self.simVars)
			k4=self.fun(t[iii]+12*h/13,y[iii,:]+1932/2197*h*k1-7200/2197*h*k2+7296/2197*h*k3,self.simVars)
			k5=self.fun(t[iii]+h,y[iii,:]+439/216*h*k1-8*h*k2+3680/513*h*k3-845/4104*h*k4,self.simVars)
			k6=self.fun(t[iii]+h/2,y[iii,:]-8/27*h*k1+2*h*k2-3544/2565*h*k3+1859/4104*h*k4-11/40*h*k5,self.simVars)

			#Solution for RKF4
			y=np.append(y,y[iii,:]+h*(25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5),axis=0)

			#Solution for RKF5
			rkf45=y[iii,:]+h*(16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6);
			#New time step
			t.append(t[iii]+h)
			error=rkf45-y[iii+1,:]
			normError=np.sqrt(np.sum(error**2))
			if normError>self.errorThresh: #If error is greater than the treshold
				kscale=np.sqrt(self.errorThresh/normError)# %Scale it
				h=h*kscale;
			elif normError<self.errorThresh*1e-3: #If error is LT the treshold
				kscale=np.sqrt(self.errorThresh*1e-3/normError)# Scale it
				h=h*kscale#

			iii+=1
			if h>self.maxStepSize:
				h=self.maxStepSize
			elif h<self.minStepSize:
				h=self.minStepSize
			
		self.resTime=t
		self.resY=y
		print('Equations have been solved!')