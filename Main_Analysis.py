import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from scipy.stats import pearsonr
from netCDF4 import Dataset

def e_s(TC):
	
	# Function to calculate saturation mixing ratio. Returns pressure in kPa
	e_s = 6.11*10**(7.5*TC/(237.5+TC))  # TEMPERATURES IN CELSIUS!!!
	return(e_s*0.1)

def calc_q_s(TC,P):	
	# Function to calculate saturation mixing ratio as a function of temperature (in celsius) and pressure in hPa
	es = e_s(TC)*10 # Transform to hPa
	return(es*0.622/(P - 0.37*es))

def calc_delta_q(TC,P):
	# Calculating derivative of the saturation specific humidity [kg h20/kg air / C]
	TC_plus = TC+.1
	TC_minus = TC-.1
	delta = (calc_q_s(TC_plus,P) - calc_q_s(TC_minus,P))/(TC_plus - TC_minus)
	return(delta)

def calc_delta_e(TC):
	# Calculating derivative of the saturation vapor pressure [hPa/C]
	TC_plus = TC+.1
	TC_minus = TC-.1
	delta = (e_s(TC_plus) - e_s(TC_minus))/(TC_plus - TC_minus)
	return(delta)

def BL_Newton():

##################################################################################
###################### SIMPLE MODEL FOR FIGURE 2 #########################
##################################################################################

	from scipy.stats import pearsonr

	########## PHYSICAL CONSTANTS

	rho_a = 1.25 		# [kg/m^3]
	L     = 2.5e06	 	# [J/kg]
	sigma = 5.67e-08	# Stephan Boltzman Constant
	T_freeze = 273.15	# Kelvin
	cp = 1003		# heat capacity of air

	##### TUNABLE PARAMETERS 

	P_s	 = 1013		# hPa
	h 	= 1000		# meters
	S_n 	= 250 		# [W/m^2] hot/dry = 350 cool/humid = 200 , norm = 250
	L_down 	= 300 		# [W/m^2]
	g_a 	= 1/50. 	# [m/s]
	g_o	= 1/900. 	# [m/s]
	g_r 	= h/(10*86400.)	# [m/s] 
	T_R 	= 10 + 273.15 	# [K]
	q_R	= 0.001		# [kg H2O / kg air] hot/dry = 0.001 cool/humid = 0.01 , norm = 0.001
	alpha = (1 + g_r/g_a)**(-1)

########### Looking across a range of soil moisture values
	N = 50
	m = np.linspace(0.01,.99,N) 	# soil moisture values
	Gs = g_o*m			# surface conductance is a linear function of soil moisture
	Gamma = rho_a*g_a*Gs/(g_a + Gs)	# combined surface parameter in ET equation

	beta = rho_a*g_r/Gamma
	beta_inv = (1 + beta)**(-1)
	Ts_master = np.zeros(N)
	theta_master = np.zeros(N)
	q_master = np.zeros(N)

	i = 0
	while i < N:

		Ts_guess = 280		# First guess in Newton's method
		F_x = 190		# innital imbalance in surface energy budget

		while abs(F_x) > 0.05:	# We'll search until the surface energy budget converges
			theta = (Ts_guess + (g_r/g_a)*T_R)*alpha
			q = (calc_q_s(Ts_guess - T_freeze,P_s) + beta[i]*q_R)*beta_inv[i]

			F_x = S_n + L_down - sigma*(Ts_guess**4) - rho_a*cp*g_a*(Ts_guess - theta) - L*Gamma[i]*(calc_q_s(Ts_guess - T_freeze,P_s) - q)
			df_dx = -4*sigma*(Ts_guess**3) - rho_a*cp*g_a*(1-alpha) - L*Gamma[i]*(1-beta_inv[i])*calc_delta_q(Ts_guess - T_freeze,P_s)
			Ts_guess = Ts_guess - (F_x/df_dx)
			
		Ts_master[i] = Ts_guess	
		theta_master[i] = (Ts_guess + (g_r/g_a)*T_R)*alpha
		q_master[i] = (calc_q_s(Ts_guess - T_freeze,P_s) + beta[i]*q_R)*beta_inv[i]

		i+=1

	Gs = Gs*86400.

	####### VPD defined as e_s(T_s) - e_r (see introduction)
	e_s_master = e_s(theta_master-T_freeze)
	e_master = 0.1*q_master*P_s/0.622 # in kPa
	VPD = (e_s_master - e_master)

	######### FITTING THE MEDLYN MODEL #####################################

	X = np.polyfit(VPD**(-0.5),Gs,deg=1)
	r_val = pearsonr(Gs,X[1] + X[0]/(VPD**0.5))
	print('Correlation with Medlyn is ' + str(r_val))

	########### FITTING THE BALL-BERRY MODEL#############################

	BB = np.polyfit((1 - (VPD/e_s_master)),Gs,deg=1)
	r_val = pearsonr(Gs,BB[1] + BB[0]*(1 - (VPD/e_s_master)))
	print('Correlation with BB is ' + str(r_val))

	########## FITTING THE Oren 99 Model ################################
	p = np.polyfit(-np.log(VPD),Gs,deg=1)
	# get correlation between fitted O99 model and the equilibrium curves
	r_val = pearsonr(Gs,p[1] -p[0]*np.log(VPD))
	print('Correlation with O99 is ' + str(r_val))
	alpha = p[0]/p[1]
	print(alpha)

	
##################################################################################	
	#################### MAKING FIGURE 
##################################################################################

#	plt.plot(m,Gs,'k')
#	plt.ylim(0,5.5)

#	fig, ax1 = plt.subplots()
#	ax1.plot(m,theta_master-T_freeze,'k')
#	ax2 = ax1.twinx()
#	ax2.tick_params(axis='y',labelcolor='r')
#	ax2.plot(m,VPD,'r')


#	plt.plot(VPD,Gs,'k',linewidth=3)
#	plt.plot(VPD,p[1] - p[0]*np.log(VPD),'C0--',linewidth=3)
#	plt.plot(VPD,BB[1] + BB[0]*(1 - VPD/e_s(theta_master-T_freeze)),'C1--',linewidth=3)
#	plt.plot(VPD,X[1] + X[0]/(VPD**0.5),'C2--',linewidth=3)
#	plt.ylim(0,100)
#	plt.xlim(1.5,5.5)
	
#	plt.savefig('Fig_2_pres1.pdf')
#	plt.show()

def EQ_with_Noise(Rad,m,T_R,q_R):

	########## PHYSICAL CONSTANTS

	rho_a = 1.25 		# [kg/m^3]
	L     = 2257000 	# [J/kg]
	sigma = 5.67e-08	# Stephan Boltzman Constant
	T_freeze = 273.15	# Kelvin
	cp = 1003		# heat capacity of air

	##### TUNABLE PARAMETERS 

	P_s	= 1013		# hPa
	h 	= 1000		# meters
	g_a 	= 1/50. 	# [m/s]
	g_o	= 1/900. 	# [m/s]
	g_r 	= h/(10*86400.)	# [m/s] 
	alpha = (1 + g_r/g_a)**(-1)

	Gs = g_o*m			# surface conductance is a linear function of soil moisture
	Gamma = rho_a*g_a*Gs/(g_a + Gs)	# combined surface parameter in ET equation

	beta = rho_a*g_r/Gamma
	beta_inv = (1 + beta)**(-1)

	Ts_guess = 280		# First guess in Newton's method
	F_x = 190		# innital imbalance in surface energy budget

	while abs(F_x) > 0.05:	# We'll search until the surface energy budget converges
		theta = (Ts_guess + (g_r/g_a)*T_R)*alpha
		q = (calc_q_s(Ts_guess - T_freeze,P_s) + beta*q_R)*beta_inv
		F_x = Rad - sigma*(Ts_guess**4) - rho_a*cp*g_a*(Ts_guess - theta) - L*Gamma*(calc_q_s(Ts_guess - T_freeze,P_s) - q)
		df_dx = -4*sigma*(Ts_guess**3) - rho_a*cp*g_a*(1-alpha) - L*Gamma*(1-beta_inv)*calc_delta_q(Ts_guess - T_freeze,P_s)
		Ts_guess = Ts_guess - (F_x/df_dx)


################## Parameters needed for Penman Monteith
			
	theta = (Ts_guess + (g_r/g_a)*T_R)*alpha
	q = (calc_q_s(Ts_guess - T_freeze,P_s) + beta*q_R)*beta_inv
	Dqa = calc_q_s(theta-T_freeze,P_s) - q
	LHF = L*Gamma*Dqa
	Delta = calc_delta_q(theta - T_freeze,P_s)
	gamma = cp/L 
	RH = q/(Dqa + q)
	VPD = e_s(theta - T_freeze)*(1 - RH)
	Rn = Rad - sigma*(Ts_guess**4)

	Gs_PM = g_a*gamma/((((Rn*Delta) + (cp*rho_a*g_a*Dqa))/LHF) - Delta - gamma)

	return(Gs_PM*86400,m,VPD)

def quantile_analysis():

	from scipy.stats import pearsonr

	NF = 1000
	Rscale = 50
	Rad_noise = np.random.normal(loc=0,scale=Rscale,size=NF)
	Rad = 650 - Rad_noise

################# CORRELATED NOISE

#	T = 285 - Rad_noise/Rscale
#	np.putmask(T,T<275,275)
#
#	q = 0.004 + Rad_noise/(Rscale*1000.)
#	np.putmask(q,q<0,0)

###################### UNCORRELATED NOISE

	T = np.random.normal(loc=280,scale=2,size=NF)
	q = np.random.normal(loc=0.005,scale = 0.0005,size=NF)
	m = np.random.normal(loc=0.5,scale=0.1,size=NF)

	np.putmask(q,q<0,0)
	np.putmask(m,m<0,0)
	np.putmask(m,m>1,1)
	np.putmask(T,T<275,275)


	VPD = np.zeros(NF)
	G_s = np.zeros(NF)
	i = 0

	while i < NF:

		Gs_val,m_val,VPD_val = EQ_with_Noise(Rad[i],m[i],T[i],q[i])		
		VPD[i] = VPD_val
		G_s[i] = Gs_val				

		i+=1


################## FOR ERROR EXPERIMENTS, STOP HERE

	return(G_s,m,VPD)

################# MAKING QUANTILE PLOT

	nquant =4
	vquantile_box = np.zeros(shape=(nquant,nquant))*np.nan
	vsig_box = np.zeros(shape=(nquant,nquant))

	mquantile_box = np.zeros(shape=(nquant,nquant))*np.nan
	msig_box = np.zeros(shape=(nquant,nquant))

############ calculate quantiles of VPD and m

	m_quants = np.zeros(nquant+1)
	V_quants = np.zeros(nquant+1)

	i = 0
	quant_list = np.linspace(0,100,nquant+1)
	while i < nquant+1:
		m_quants[i] = np.nanpercentile(m,q=quant_list[i])
		V_quants[i] = np.nanpercentile(VPD,q=quant_list[i])
		i+=1

	i = 0
	while i < nquant:
		j = 0
		while j < nquant:

			my_slice = np.where((VPD<V_quants[i+1]) & (VPD>V_quants[i]) & (m<m_quants[j+1]) & (m>m_quants[j]))
			indicies = my_slice[0]

			gs_slice = G_s[indicies]
			m_slice = m[indicies]
			v_slice = VPD[indicies]

			if len(gs_slice) > 3:
				vslop,inter = np.polyfit(v_slice,gs_slice,deg=1)
				rv,pv = pearsonr(v_slice,gs_slice)

				mslop,inter = np.polyfit(m_slice,gs_slice,deg=1)
				rm,pm = pearsonr(m_slice,gs_slice)

				if pv < 0.01/(nquant**2):
					vsig_box[i,j] = 1

				if pm < 0.01/(nquant**2):
					msig_box[i,j] = 1

				vquantile_box[i,j] = vslop		
				mquantile_box[i,j] = mslop

			j+=1	
		i+=1


	X = (m_quants[1:] + m_quants[:nquant])/2.
	Y = (V_quants[1:] + V_quants[:nquant])/2.

	plt.subplot(2,1,1)
	plt.scatter(m,G_s)
	plt.subplot(2,1,2)
	plt.scatter(VPD,G_s)
	plt.show()

	X = np.linspace(5,95,nquant)
	Y = np.linspace(5,95,nquant)

############## FOR SOIL MOISTURE	
	mmin  = 0
	mmax = np.nanmax(mquantile_box)	
	mcmap = 'Greens'
############## FOR VPD
	vmax = -np.nanmin(vquantile_box)
	vmin = np.nanmin(vquantile_box)
	cmap = 'BrBG'

	v_masked = np.ma.masked_where(vsig_box,vsig_box>0)
	m_masked = np.ma.masked_where(msig_box,msig_box>0)

	plt.figure(figsize=(14,6))
	plt.subplot(1,2,1)	
	plt.pcolormesh(X,Y,mquantile_box,cmap=mcmap,vmin=mmin,vmax=mmax)
	plt.colorbar()
	plt.pcolor(X,Y,m_masked,hatch='/',alpha=0)

	plt.subplot(1,2,2)
	plt.pcolormesh(X,Y,vquantile_box,cmap=cmap,vmin=vmin,vmax=vmax)
	plt.colorbar()
	plt.pcolor(X,Y,v_masked,hatch='/',alpha=0)
	plt.savefig('sensitivity.pdf')
	plt.show()

def EQ_EXPS():

	#### MAKE EXPERIMENTS WITH A PRESRIBED AMOUNT OF NOISE ADDED POST FACTO

	G_s,m,VPD = quantile_analysis()
	N = len(G_s)

	V_noise = 0
	m_noise = 7

	VPDnudge = (rand.rand(N) - 0.5)*np.nanstd(VPD)*V_noise + VPD
	mnudge = (rand.rand(N) - 0.5)*np.nanstd(m)*m_noise + m

	from scipy.stats import pearsonr

	np.putmask(mnudge,mnudge<0,0)
	np.putmask(mnudge,mnudge>1,1)

	print(pearsonr(mnudge,m))
	print(pearsonr(VPDnudge,VPD))

	np.putmask(VPDnudge,VPDnudge<np.nanmin(VPD),np.nanmin(VPD))

	p = np.polyfit(-np.log(VPDnudge),G_s,deg=1)
	p_m = np.polyfit(mnudge,G_s,deg=1)
	
	m_lin = np.linspace(0,1,20)
	VPD_lin = np.logspace(-1,1,20)
	rstuff_m = pearsonr(G_s,p_m[1] + p_m[0]*mnudge)
	print('SOIL MOISTURE R is '+str(rstuff_m[0]))
	print(rstuff_m[1])

	rstuff_VPD = pearsonr(G_s,p[1] - p[0]*np.log(VPDnudge))
	print('VPD R is ' +str(rstuff_VPD[0]))
	print(rstuff_VPD[1])
	plt.figure(figsize=(10,4))
	plt.subplot(1,2,1)
	plt.plot(mnudge,G_s,'k.')
	plt.plot(m_lin,p_m[1] + p_m[0]*m_lin,'r--')
	plt.ylim(-25,np.nanmax(G_s)+50)
	plt.xlim(-0.1,1.1)
	plt.subplot(1,2,2)
	plt.plot(VPDnudge,G_s,'k.')
	plt.plot(VPD_lin,p[1] - p[0]*np.log(VPD_lin),'r--')
	plt.ylim(-25,np.nanmax(G_s)+50)
	plt.xlim(-0.5,10)
#	plt.savefig('EXPs_Vnudge.pdf')
	plt.show()

def read_bville():
	path = '/Users/lucaszeppetello/Desktop/Projects/Surface_Conductance/Data/'
	f = open(path+'bflatEdCovTT.txt','r')
	line = f.readlines()[1:]

	Nobs = len(line)

	T = []
	E = []	
	RH = [] 
	P = [] 
	OLR = []
	Net_Rad = [] 
	speed = []


	i = 0

	while i < Nobs:
	
		this_line = line[i]
		mylist = this_line.split(',')

		Net_Rad.append(float(mylist[1])) 	# in W/m^2
		P.append(float(mylist[2])*10)		# in hPa		
		T.append(float(mylist[3])) 		# in C
		RH.append(float(mylist[5])/100.)	# in fraction
		speed.append(float(mylist[6])*5/18.)	# in m/s
		OLR.append(float(mylist[16])) 		# in W/m^2
		E.append(float(mylist[20])/(30*60)) 	# kg/m^2/s

		i +=1

	return(Net_Rad,P,T,RH,speed,OLR,E)

def read_Garcia(site):

	path = '/Users/lucaszeppetello/Desktop/Projects/Surface_Conductance/Data/'
	f = open(path+'Playa'+site+'_30min.csv')
	lines = f.readlines()

	i = 3
	Nobs = len(lines)-i

	T = []
	LHF = []	
	RH = [] 
	Net_Rad = [] 
	speed = []
	H = []
	G = []
	e = []

	while i < Nobs:
		myline = lines[i].split(",")
		T.append(float(myline[7])) # Celsius
		LHF.append(float(myline[2])) #kg/m^2/s
		RH.append(float(myline[12])/100.) # fraction
		Net_Rad.append(float(myline[4])) # W/m^2
		G.append(float(myline[5]))  # W/m^2	
		speed.append(float(myline[9]))	  #m/s
		H.append(float(myline[3])) # W/m^2
		e.append(float(myline[11])) # kPa
		i+=1
	
	return(Net_Rad,H,T,RH,speed,LHF,G,e)
		
def calc_ga(u):

	# Parameters
	k = 0.41		# Von Karman constant
	h = 0.1			# Canopy height [m] 
	zm = 2			# Measurement height [m]
	zd = 0.67*h		# Zero plane displacement [m]
	zo = 0.1*h		# Momentum roughness length [m]
	Z_combine = (zm - zd)/zo
	# calculation

	g_a = u*(k**2)/((np.log(Z_combine))**2) # See Novick et al. 2016 (originally from Campbell & Norman 1998)
	return(g_a)

def calc_gs_method_I():

	# This is a direct inversion of the prosaic ET formula that relies on an estimate of the surface temperature via OLR data
	# Reading data from the Bonneville salt flats
	Net_Rad,P,T_a,RH,speed,OLR,E = read_bville()

	T_a = np.asarray(T_a)
	P = np.asarray(P)
	RH = np.asarray(RH)
	u = np.asarray(speed)
	OLR = np.asarray(OLR)
	E = np.asarray(E)

	######### Physical Constants

	rho_a = 1.2 		# kg/m^3	
	sigma = 5.67e-08	# Stefan-Boltzmann [W/m^2/K^4]
	L = 2.5e06		# Latent Enthalpy of Vaporization [J/kg H2O]
	LHF = L*E
	
	# calculating g_a
	g_a = calc_ga(u)

	########## Calculating surface temperature 

	T_s = ((OLR/sigma)**0.25) -273.15	# Assuming surface emissivity of 1 [Celsius]

	########## Humidity deficit

	q_s = calc_q_s(T_s,P)			# Calculating surface specific humidity - assumed saturated
	q_s_a = calc_q_s(T_a,P)			# Assumes 2m pressure = surface pressure
	q_a = RH*q_s_a				# Calculating 2m specific humidity
	grad_q = q_s - q_a			# Difference of surface and 2m specific humidity 

	#### VPD 

	VPD = e_s(T_a)*(1 - RH)			# atmospheric VPD [kPa]

	######## Calculation Time

	g_s = np.ones(len(q_a))*np.nan
	min_LHF = 20				# Screening out small values of Latent Heat Flux to get daytime values [W/m^2]

	i = 0
	while i < len(VPD):
		if LHF[i] > min_LHF:
			g_s[i] = g_a[i]/((g_a[i]*rho_a*grad_q[i]/E[i]) - 1)
		i+=1

	return(VPD,g_s)

def calc_gs_PM_Nevada(site):

	Net_Rad,H,T_a,RH,speed,LHF,G,e = read_Garcia(site)
	Net_Rad = np.asarray(Net_Rad)
	T_a = np.asarray(T_a)
	RH = np.asarray(RH)
	u = np.asarray(speed)
	LHF = np.asarray(LHF)
	H = np.asarray(H)
	G = np.asarray(G)
	Rn = Net_Rad - G
	e = np.asarray(e)
	Nobs = len(e)

	LHF[LHF > Rn] = -9999	

	######### Physical Constants

	rho_a = 1.2 		# kg/m^3	
	cp = 1003		# Specific Heat of air [J/kg air / K]
	L = 2.5e06		# Latent Enthalpy of vaporization
	gamma = cp/L		# psychrometic constant [kg H2O/kg air / K]

	########## calculating aerodynamic conductance

	g_a = calc_ga(u)
	sat_e = e_s(T_a) 		# Saturated Vapor Pressure in kPa
	VPD = sat_e - e 		# VPD in kPa
	Delta_e = calc_delta_e(T_a) 	# in kPa per C

	g_s = np.ones(Nobs)*np.nan
	i = 0
	min_LHF = 20
	min_VPD = 1

	while i < Nobs:
		if LHF[i] > min_LHF and VPD[i] > min_VPD:
			g_s[i] = g_a[i]*gamma/((((Rn[i]*Delta_e[i]) + (cp*rho_a*g_a[i]*VPD[i]))/LHF[i]) - Delta_e[i] - gamma)
		i+=1

	return(VPD,g_s)

def calc_gs_PM_Bville():
	# This is an inversion of the Penman Monteith Equation
	# For the Boneville site where surface pressure measurements are available

	Net_Rad,P,T_a,RH,speed,OLR,E = read_bville()
	Rn = np.asarray(Net_Rad)
	T_a = np.asarray(T_a)
	P = np.asarray(P)
	RH = np.asarray(RH)
	u = np.asarray(speed)
	OLR = np.asarray(OLR)
	E = np.asarray(E)

	######### Physical Constants

	rho_a = 1.2 		# kg/m^3	
	L = 2.5e06		# Latent Enthalpy of Vaporization [J/kg H2O]
	cp = 1003		# Specific Heat of air [J/kg air / K]
	gamma = cp/L		# psychrometic constant [kg H2O/kg air / K]
	LHF = L*E		# Latent Heat Flux [W/m^2]

	########## calculating aerodynamic conductance

	g_a = calc_ga(u) # Campbell & Norman 1998

	##### Humidity Gradient

	q_s_a = calc_q_s(T_a,P)			# Assumes 2m pressure = surface pressure
	q_a = RH*q_s_a				# Calculating 2m specific humidity
	D_q = q_s_a - q_a			# Difference of surface and 2m specific humidity 

	####### Delta

	Delta = calc_delta_q(T_a,P)			# Slope of q_s/T curve evaluated at air temperature

	###### VPD

	VPD = e_s(T_a)*(1 - RH)

	##### Calculation Time
	g_s = np.ones(len(Delta))*np.nan
	min_LHF = 20				# Screening parameter
	i = 0
	while i < len(g_s):
		if LHF[i] > min_LHF:
			g_s[i] = g_a[i]*gamma/((((Rn[i]*Delta[i]) + (cp*rho_a*g_a[i]*D_q[i]))/LHF[i]) - Delta[i] - gamma)
		i+=1

	return(VPD,g_s)
	

def GS_vs_VPD(VPD,gs):
	from scipy.stats import pearsonr

	idx = np.isfinite(VPD) & np.isfinite(gs)

	print("n = " +str(len(VPD[idx])))

	p = np.polyfit(-np.log(VPD[idx]),gs[idx],deg=1)
	slope = p[0]/p[1]  # SLOPE parameter from O99

	print(slope)

	rstuff = pearsonr(gs[idx],p[1] -p[0]*np.log(VPD[idx]))
	print(rstuff)      # Correlation with O99 Model

############# PLOTTING #########################################################

	VPD_lin = np.logspace(-1,2,20)
	plt.plot(VPD[idx],gs[idx]*86400,'k.',markersize=7)
	plt.xlim(0,np.nanmax(VPD)+0.5)
#	plt.ylim(-1,10)
	plt.ylim(-50,np.nanmax(gs*86400) + 50)
	plt.plot(VPD_lin,86400*(p[1] - p[0]*np.log(VPD_lin)),'r')
	plt.savefig('Bville_PM.pdf')
	plt.show()

def Random_G_s():

	N = 1000

	T_rand = np.random.normal(loc=295,scale=5,size=N)
	q_rand = np.random.normal(loc=0.001,scale=0.0005,size=N)
	LHF_rand = np.random.normal(loc=100,scale=20,size=N)
	Rn_rand = np.random.normal(300,scale=30,size=N)

	np.putmask(T_rand,T_rand<275,275)
	np.putmask(LHF_rand,LHF_rand<20,20)
	np.putmask(q_rand,q_rand<0.0001,0.0001)
	np.putmask(Rn_rand,Rn_rand<50,50)

	D_qa = calc_q_s(T_rand-273.15,1013) - q_rand
	RH = q_rand/calc_q_s(T_rand-273.15,1013)

	cp = 1003
	rho_a = 1.25
	L = 2.5e06		# Latent Enthalpy of Vaporization [J/kg H2O]
	E = LHF_rand/L

	Delta = calc_delta_q(T_rand-273.15,1013)	# calculate delta at P_s = 1013 hPa

	g_a = .02 # aero resistance
	gamma = cp/L	# psychrometric constant

	######### PENMAN MONTEITH METHOD
	g_s = 86400*g_a*gamma/((((Rn_rand*Delta) + (cp*rho_a*g_a*D_qa))/LHF_rand) - Delta - gamma)
	######### "DIRECT VERSION"
	g_s_D = 86400*g_a*(((g_a*rho_a*D_qa/E) - 1)**-1)
	
	VPD = e_s(T_rand-273.15)*(1-RH)

	p = np.polyfit(-np.log(VPD),g_s,deg=1)
	p_D = np.polyfit(-np.log(VPD),g_s_D,deg=1)

	print(pearsonr(np.log(VPD),g_s))
	print(pearsonr(np.log(VPD),g_s_D))
	print(p[0]/p[1])
	print(p_D[0]/p_D[1])

	if np.nanmax(g_s) > np.nanmax(g_s_D):
		maxmax = np.nanmax(g_s)
	else:
		maxmax = np.nanmax(g_s_D)

	VPD_lin = np.logspace(-1,2,20)


	######### PLOTTING

	plt.figure(figsize=(10,4))
	plt.subplot(1,2,2)
	plt.plot(VPD_lin,(p[1] - p[0]*np.log(VPD_lin)),'r--')
	plt.scatter(VPD,g_s,c='k')
	plt.xlim(0,6)
	plt.ylim(0,maxmax*1.1)
	plt.subplot(1,2,1)
	plt.plot(VPD_lin,(p_D[1] - p_D[0]*np.log(VPD_lin)),'r--')
	plt.scatter(VPD,g_s_D,c='k')
	plt.xlim(0,6)
	plt.ylim(0,maxmax*1.1)
	plt.savefig('Random_g_s.pdf')
	plt.show()

############## Observational Figures ###############################

#v,g = calc_gs_PM_Nevada('1')
#v,g = calc_gs_PM_Nevada('2')
#v,g = calc_gs_method_I()
#v,g = calc_gs_PM_Bville()
#GS_vs_VPD(v,g)

############## Three Points Figure ##############################
#BL_Newton()

############### MAKING Fu et al. 2022-like Figure  #############################3
#quantile_analysis()

############ Experiments with noise in the BL Model
#EQ_EXPS()

############## Supplement
Random_G_s()
