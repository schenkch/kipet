#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Aspirin Example
# find the model details in the Kipet documentation
#		\frac{dZ_aa}{dt} = -r_0-r_1-r_3-\frac{\dot{v}}{V}*Z_aa
# 	    	\frac{dZ_ha}{dt} = r_0+r_1+r_2+2r_3-\frac{\dot{v}}{V}*Z_ha
#        \frac{dZ_asaa}{dt} = r_1-r_2-\frac{\dot{v}}{V}*Z_asaa
#        \frac{dZ_h2o}{dt} = -r_2-r_3+\frac{f}{V}*C_h2o^in-\frac{\dot{v}}{V}*Z_asaa

#        \frac{dm_{sa}}{dt} = -M_{sa}*V*r_d
#        \frac{dm_{asa}}{dt} = -M_{asa}*V*r_c
#        \frac{dV}{dt} = V*\sum_i^{ns}\upsilon_i*(\sum_j^{6}\gamma_i*r_j+\epsilon_i*\frac{f}{V}*C_h2o^in)

#        r_0 = k_0*Z_sa*Z_aa
#        r_1 = k_1*Z_asa*Z_aa
#        r_2 = k_2*Z_asaa*Z_h2o
#        r_3 = k_3*Z_aa*Z_h2o
#        r_d = k_d*(Z_sa^{sat}-Z_sa)^d
#        r_c = k_c*(max(Z_asa-Z_sa^{sat}))^c

from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.fe_factory import *
from kipet.library.FESimulator import *
#from pyomo.core.kernel.expr import exp
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pprint

if __name__ == "__main__":
    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
 
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    traj =  os.path.join(dataDirectory,'extra_states.txt')

    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    conc =  os.path.join(dataDirectory,'concentrations.txt')    
    
    fixed_traj = read_absorption_data_from_txt(traj)
    C = read_absorption_data_from_txt(conc)
    
    meas_times=sorted(C.index)
    #print(meas_times)
    # How many measurement times are there
    nfe_x = len(meas_times)
    #print(nfe_x)
    # create template model 
    builder = TemplateBuilder()    

    # components
    components = dict()
    components['SA'] = 1.0714               # Salicitilc acid
    components['AA'] = 9.3828               # Acetic anhydride
    components['ASA'] = 0.0177              # Acetylsalicylic acid
    components['HA'] = 0.0177               # Acetic acid
    components['ASAA'] = 0.000015           # Acetylsalicylic anhydride
    components['H2O'] = 0.0                 # water

    builder.add_mixture_component(components)

    # add parameters
    params = dict()
    params['k0'] = 0.0360309
    params['k1'] = 0.1596062
    params['k2'] = 6.8032345
    params['k3'] = 1.8028763
    params['kd'] = 7.1108682
    params['kc'] = 0.7566864
    params['Csa'] = 2.06269996

    builder.add_parameter(params)

    # add additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0202
    extra_states['Masa'] = 0.0
    extra_states['Msa'] = 9.537
    
    builder.add_complementary_state_variable(extra_states)

    algebraics = ['f','Csat','r0','r1','r2','r3','r4','r5','v_sum']

    builder.add_algebraic_variable(algebraics)

    #remove the f and Csat algebraic variables and fix them as time-dependent parameters
    #now we fix the non-variable for the fe_factory, f and Csat

    gammas = dict()
    gammas['SA']=    [-1, 0, 0, 0, 1, 0]
    gammas['AA']=    [-1,-1, 0,-1, 0, 0]
    gammas['ASA']=   [ 1,-1, 1, 0, 0,-1]
    gammas['HA']=    [ 1, 1, 1, 2, 0, 0]
    gammas['ASAA']=  [ 0, 1,-1, 0, 0, 0]
    gammas['H2O']=   [ 0, 0,-1,-1, 0, 0]


    epsilon = dict()
    epsilon['SA']= 0.0
    epsilon['AA']= 0.0
    epsilon['ASA']= 0.0
    epsilon['HA']= 0.0
    epsilon['ASAA']= 0.0
    epsilon['H2O']= 1.0
    
    partial_vol = dict()
    partial_vol['SA'] = 0.0952552311614
    partial_vol['AA'] = 0.101672206869
    partial_vol['ASA'] = 0.132335206093
    partial_vol['HA'] = 0.060320218688
    partial_vol['ASAA'] = 0.186550717015
    partial_vol['H2O'] = 0.0883603912169
    
    def rule_algebraics(m,t):
        r = list()
        r.append(m.Y[t,'r0']-m.P['k0']*m.Z[t,'SA']*m.Z[t,'AA'])
        r.append(m.Y[t,'r1']-m.P['k1']*m.Z[t,'ASA']*m.Z[t,'AA'])
        r.append(m.Y[t,'r2']-m.P['k2']*m.Z[t,'ASAA']*m.Z[t,'H2O'])
        r.append(m.Y[t,'r3']-m.P['k3']*m.Z[t,'AA']*m.Z[t,'H2O'])

        # dissolution rate/1e-4    5e-2
        step = 1.0/(1.0 + exp(-m.X[t,'Msa']/5e-2))
        rd = m.P['kd']*(m.P['Csa']-m.Z[t,'SA']+1e-6)**1.90*step
        r.append(m.Y[t,'r4']-rd)
        #r.append(m.Y[t,'r4'])
        
        # crystalization rate
        diff = m.Z[t,'ASA'] - m.Y[t,'Csat']
        rc = 0.3950206559*m.P['kc']*(diff+((diff)**2+1e-6)**0.5)**1.34
        r.append(m.Y[t,'r5']-rc)

        Cin = 39.1
        v_sum = 0.0
        V = m.X[t,'V']
        f = m.Y[t,'f']
        for c in m.mixture_components:
            v_sum += partial_vol[c]*(sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin)
        r.append(m.Y[t,'v_sum']-v_sum)

        return r

    builder.set_algebraics_rule(rule_algebraics)
    
    def rule_odes(m,t):
        exprs = dict()

        V = m.X[t,'V']
        f = m.Y[t,'f']
        Cin = 41.4
        # volume balance
        vol_sum = 0.0
        for c in m.mixture_components:
            vol_sum += partial_vol[c]*(sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin)
        exprs['V'] = V*m.Y[t,'v_sum']

        # mass balances
        for c in m.mixture_components:
            exprs[c] = sum(gammas[c][j]*m.Y[t,'r{}'.format(j)] for j in range(6))+ epsilon[c]*f/V*Cin - m.Y[t,'v_sum']*m.Z[t,c]

        exprs['Masa'] = 180.157*V*m.Y[t,'r5']
        exprs['Msa'] = -138.121*V*m.Y[t,'r4']
        return exprs




    builder.set_odes_rule(rule_odes)

    model = builder.create_pyomo_model(0.0,210.5257)    
 #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     
    fe_x_list = [ii for ii in range(1, nfe_x + 1)]
    model.fe_x_i = Set(initialize=fe_x_list)
    
    sim = FESimulator(model)
    
    # defines the discrete points wanted in the concentration profile(nfe_x)
    sim.apply_discretization('dae.collocation', nfe=200, ncp=3, scheme='LAGRANGE-RADAU')
    fe_l = sim.model.time.get_finite_elements()
    fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]
    nfe = len(fe_list)  #: Create a list with the step-size
    print(nfe)
    
    for i in sim.model.X.itervalues():
        idx = i.index()
        if idx[1] in ['Msa']:
            i.setlb(0)
        else:
            i.setlb(0)
            
    inputs_sub = {}
    
    inputs_sub['Y'] = ['f', 'Csat']

    sim.fix_from_trajectory('Y','Csat',fixed_traj)
    sim.fix_from_trajectory('Y','f',fixed_traj)

    init = sim.call_fe_factory(inputs_sub)
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #=========================================================================
  
            
    #sim.fix_from_trajectory('Y','Csat',fixed_traj)
    #sim.fix_from_trajectory('Y','f',fixed_traj)

    #for i in sim.model.X.itervalues():
    #    idx = i.index()
    #    if idx[1] in ['Msa', 'Masa']:
    #        i.setlb(-0.05)
    #    else:
    #        i.setlb(0)

    #for i in sim.model.Z.itervalues():
    #    i.setlb(0)
    #    idx = i.index()
    #    if idx[1] == 'SA':
    #        i.setub(2.06269996)


    options = {'halt_on_ampl_error' :'yes',
               'bound_push': 1e-06,
               'print_user_options': 'yes',
               "max_iter": 1}
    results = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)
    if with_plots:
        # display concentration results    
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        C.plot()
        
        plt.figure()
        
        results.Y['Csat'].plot.line()
        plt.plot(fixed_traj['Csat'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("Csat")
        plt.title("Saturation Concentration")
        
        plt.figure()
        
        results.X['V'].plot.line()
        plt.plot(fixed_traj['V'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("volumne (L)")
        plt.title("Volume Profile")

        plt.figure()
        
        results.X['Msa'].plot.line()
        plt.plot(fixed_traj['Msa'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("m_dot (g)")
        plt.title("Msa Profile")

        plt.figure()
        results.Y['f'].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("flow (K)")
        plt.title("Inlet flow Profile")

        plt.figure()
        results.X['Masa'].plot.line()
        plt.plot(fixed_traj['Masa'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("m_dot (g)")
        plt.title("Masa Profile")
        
plt.show()