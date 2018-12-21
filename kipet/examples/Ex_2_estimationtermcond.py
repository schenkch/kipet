
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknown variances of spectral data using fe-factory model with inputs
#
#		\frac{dZ_a}{dt} = -k_1*Z_a - (Vdot/V)*Z_a	                                                        Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2(T)*Z_b	- (Vdot/V)*Z_b	+ {1. m_Badd/V/1.1 for t<1.1s or 2. 0 for t>1.1s} Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2(T)*Z_b - (Vdot/V)*Z_c	                                                        Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilderaddbc2 import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.FESimulator import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
        
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-2,'B':0,'C':0}
    builder.add_mixture_component(components)
    endcomponents = {'A': 5e-7}
    builder.add_term_component(endcomponents)
    
    algebraics=['1','2','3']
    builder.add_algebraic_variable(algebraics)
    
    # Load Temp data:
    dataDirectory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe()))), 'data_sets'))
    
    
    params = dict()
    params['k1'] = 1.0
    params['k2'] = 0.2265
   # params['E'] = 2.

    builder.add_parameter(params)
    
    # add additional state variables
    #extra_states = dict()
    #extra_states['V'] = 0.0629418
      
    #builder.add_complementary_state_variable(extra_states)

    # stoichiometric coefficients
    gammas = dict()
    gammas['A'] = [-1, 0]
    gammas['B'] = [1, -1]
    gammas['C'] = [0, 1]
    
    # define system of DAEs

    def rule_algebraics(m, t):
        r = list()
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A'])
        r.append(m.Y[t, '2'] - m.P['k2'] * m.Z[t, 'B'])
        return r
    
    builder.set_algebraics_rule(rule_algebraics)

    def rule_odes(m, t):
        exprs = dict()
        for c in m.mixture_components:
            exprs[c] = gammas[c][0] * m.Y[t, '1'] + gammas[c][1] * m.Y[t, '2']  #- exprs['V'] / V * m.Z[t, c]

        return exprs
    
    builder.set_odes_rule(rule_odes)
     
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)
    
    builder.add_spectral_data(D_frame)

       
    opt_model = builder.create_pyomo_model(0.0,10.0)
    
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #========================================================================
    sim = PyomoSimulator(opt_model)

    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    #: now the final run, just as before
    # simulate
    options = {}
    # options = { 'acceptable_constr_viol_tol': 0.1, 'constr_viol_tol':0.1,# 'required_infeasibility_reduction':0.999999,
    #                'print_user_options': 'yes'}#,
    results = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)
    if with_plots:
        
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")# Without Noise")
        plt.show()
        #results.Y[0].plot.line()
        results.Y['1'].plot.line(legend=True)
        results.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()
        
        #results.X.plot.line(legend=True)
        #plt.xlabel("time (s)")
        #plt.ylabel("Volume (L)")
        #plt.title("total volume")
        #plt.show()
    sys.exit()
    #=========================================================================
    #USER INPUT SECTION - Variance Estimation
    #========================================================================
    model=builder.create_pyomo_model(0.0,10.0)
    #Now introduce parameters as non fixed
    model.del_component(params)
    builder.add_parameter('k1',init=1.0,bounds=(0.0,5.0))
    builder.add_parameter('k2',init=0.2265,bounds=(0.,5.))#0.2265)
    #builder.add_parameter('E',2.)#init=2.,bounds=(0.,10.))
    model = builder.create_pyomo_model(0, 10)
    v_estimator = VarianceEstimator(model)
    v_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    v_estimator.initialize_from_trajectory('Z', results.Z)
    v_estimator.initialize_from_trajectory('S', results.S)
    
    options = {}
    options['bound_push'] = 1e-8
    options['mu_init'] = 1e-7

    A_set = [l for i, l in enumerate(model.meas_lambdas) if (i % 8 == 0)]

    # #: this will take a sweet-long time to solve.
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-5,
                                            max_iter=15,
                                            subset_lambdas=A_set#,
                                            #inputs_sub=inputs_sub,
                                            #trajectories=trajs,
                                            #jump=True,
                                            #jump_times=jump_times,
                                            #jump_states=jump_states,
                                           ##fixedy=True,
                                            ##fixedtraj=True,
                                            ##yfix=yfix,
                                            ##yfixtraj=yfixtraj,
                                            #feed_times=feed_times
                                            )
    
    print("\nThe estimated variances are:\n")
    
    for k, v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    sigmas = results_variances.sigma_sq

    if with_plots:
        # display concentration results
        results.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile with Noise")

        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile Without Noise")
        plt.show()
        #results.Y[0].plot.line()
        results.Y['1'].plot.line(legend=True)
        results.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()
        
        #results.X.plot.line(legend=True)
        #plt.xlabel("time (s)")
        #plt.ylabel("Volume (L)")
        #plt.title("total volume")
        #plt.show()

    #=========================================================================
    #USER INPUT SECTION - Parameter Estimation
    #========================================================================
    #In case the variances are known, they can also be fixed instead of running the Variance Estimation.
    #sigmas={'C': 1e-10,
            #'B': 1e-10,
            #'A': 1e-10,
            #'device': 1e-10}
    model =builder.create_pyomo_model(0.0,10)
    #Now introduce parameters as non fixed
    model.del_component(params)
    builder.add_parameter('k1',init=0.9,bounds=(0.0,5.0))
    builder.add_parameter('k2',init=0.2,bounds=(0.,5.))#0.2265)
    #builder.add_parameter('E',2.)#init=2.,bounds=(0.,10.))
    
    model =builder.create_pyomo_model(0.0,10)
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    # Certain problems may require initializations and scaling and these can be provided from the
    # variance estimation step. This is optional.
    p_estimator.initialize_from_trajectory('Z', results_variances.Z)
    p_estimator.initialize_from_trajectory('S', results_variances.S)
    p_estimator.initialize_from_trajectory('dZdt', results_variances.dZdt)
    p_estimator.initialize_from_trajectory('C', results_variances.C)

    # Again we provide options for the solver
    options = dict()

    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('k_aug',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        with_d_vars=True,
                                        covariance=True#,
                                        #inputs_sub=inputs_sub,
                                        #trajectories=trajs,
                                        #jump=True,
                                        #jump_times=jump_times,
                                        #jump_states=jump_states,
                                        ##fixedy=True,
                                        ##fixedtraj=True,
                                        ##yfix=yfix,
                                        ##yfixtraj=yfixtraj,
                                        #feed_times=feed_times
                                        )

    # And display the results
    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)
    
    ####Goodness of fit################
    St = np.transpose(results_pyomo.S)
    resC = results_pyomo.C
    prodCSt = resC.dot(St)
    lack1 = D_frame.sub(prodCSt, fill_value=0)
    #print(lack1.shape)
    np.savetxt("D-CStransppreproc.out", lack1, delimiter=',')
    lack1 = lack1.values
    lackf = np.linalg.norm(abs(lack1), ord='fro') 
    print('The lack of fit is:', lackf)
    print('Shape D:', D_frame.shape)
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile with Noise")

        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()
        
        results_pyomo.Y['1'].plot.line(legend=True)
        results_pyomo.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        plt.show()
        
        plt.plot(np.transpose(prodCSt))
        plt.legend
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("C$S^T$")
        plt.show()

        transpD_frame = np.transpose(D_frame)
        plt.plot(transpD_frame)
        plt.legend
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("D")
        plt.show()
