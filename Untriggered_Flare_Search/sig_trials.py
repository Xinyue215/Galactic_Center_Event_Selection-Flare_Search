#!/usr/bin/env python

import numpy as np
import csky as cy
from csky import hyp
import argparse
import getpass              
                            

username = getpass.getuser() 
sig_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/trials/sig_trials')

ana_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/GC')
ana = cy.get_analysis(cy.selections.repo, cy.selections.GCDataSpecs.GC_2011_2022, dir = ana_dir, min_sigma = 0)

for a in ana.anas:
    a.bg_space_param = cy.pdf.BgAzimuthSinDecParameterization(a.bg_data, smooth=(1,4), hkw=a.kw_space_bg['hkw'])


source_ra, source_dec = 4.64964433, -0.50503147
src = cy.sources(source_ra, source_dec)


def do_signal_trials(dt, nsig, gamma, label, N, t0, threshold, cut_n_sigma, cpus):

    conf_gauss = {                                                     
        'time': 'utf',                                                 
        'seeder': cy.seeding.GaussianUTFSeeder(threshold=threshold),  
        'fitter_args': dict(_log_params='dt', _fmin_method='Minuit'),  
        'concat_evs': True,                                            
        'cut_n_sigma': cut_n_sigma,                                   
        'update_bg': True,                                             
        'extra_keep': ['energy'],                                      
        'rates_by' : 'livetime'                                        
    }                                                                  


    inj_conf = {
    'src' : src,
    'flux' : cy.hyp.PowerLawFlux(gamma),
    'box' : False,
    'sig' : 'tw',
    't0' : t0, 
    'dt' : dt
    }
    
    # get trial runner
    tr = cy.get_trial_runner(
        src=src, 
        conf = conf_gauss, 
        inj_conf=inj_conf, 
        ana=ana, 
        dt_max = ana[0].livetime/86400., 
        use_grl=True, #good run list
        mp_cpus=cpus, 
        TRUTH = False,
        )
    # run trials
    trials = tr.get_many_fits(N, nsig, poisson=True, logging=True)
    # save to disk
    out_dir = cy.utils.ensure_dir((sig_dir + f'/gamma/{gamma}/dt/{dt}/n_sig/{nsig}/'))
    filename = out_dir + f'trial_{label}.npy'
    print('->', filename)
    # notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
    np.save(filename, trials.as_array)


   

if __name__ == "__main__":                                                 
    parser=argparse.ArgumentParser(description='Optional Arguments')       
    parser.add_argument('--dt', type = float, default=1.0)               
    parser.add_argument('--nsig', type = float, default=10.0)
    parser.add_argument('--gamma', type = float, default=2.7)
    parser.add_argument('--label', type = int, default = 1)  
    parser.add_argument('--N', type = int, default = 1)        
    parser.add_argument('--t0', type = int, default = 57570)
    parser.add_argument('--threshold', type = int, default=1000)
    parser.add_argument('--cut_n_sigma', type = int, default=5)
    parser.add_argument('--cpus', type = int, default=10)

    args= parser.parse_args()                                              
    do_signal_trials(args.dt,  args.nsig, args.gamma, args.label, args.N, args.t0, args.threshold, args.cut_n_sigma, args.cpus)
 
