#!/usr/bin/env python

import numpy as np
import csky as cy
import argparse
import getpass
from importlib import reload
reload(cy.llh)           
                            
username = getpass.getuser()
bg_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/trials/bg_trials/ts_no_prior')

ana_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/GC')
ana = cy.get_analysis(cy.selections.repo, cy.selections.GCDataSpecs.GC_2011_2022, dir = ana_dir, min_sigma = 0)


for a in ana.anas:
    a.bg_space_param = cy.pdf.BgAzimuthSinDecParameterization(a.bg_data, smooth=(1,4), hkw=a.kw_space_bg['hkw'])

source_ra, source_dec = 4.64964433, -0.50503147
src = cy.sources(source_ra, source_dec)


# build bg trials

def do_background_trials(N,label, threshold, cut_n_sigma, cpus, ts_no_prior):
    conf_gauss = {
        'time': 'utf',
        'seeder': cy.seeding.GaussianUTFSeeder(threshold=threshold),
        'fitter_args': dict(_log_params='dt', _fmin_method='Minuit', _ts_min=-10000),
        'concat_evs': True,
        'cut_n_sigma': cut_n_sigma,
        'update_bg': True,
        'extra_keep': ['energy'],
        'rates_by' : 'livetime'
    }


    # get trial runner
    tr = cy.get_trial_runner(src=src, conf = conf_gauss, ana=ana, mp_cpus=cpus, dt_max = (ana[0].livetime/86400)/2, TRUTH = False, use_grl = True)
    if not ts_no_prior:
        # run trials
        trials = tr.get_many_fits(N, logging=True)
        filename = bg_dir + '/trial_negTS_{}.npy'.format(label)
        print('->', filename)
        # notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
        np.save(filename, trials.as_array)
    else: 
        tss = []
        params = []
        tss_no_prior = []
        for i in range(N):
            trials = tr.get_one_fit(logging=True, _full_output=True)
            tss.append(trials[0])
            params.append(trials[1])
            tss_no_prior.append(trials[-1]['ts_no_prior'])
        np.save(bg_dir + f'/tss_{label}.npy', tss)
        np.save(bg_dir + f'/params_{label}.npy', params)
        np.save(bg_dir + f'/tss_no_prior_{label}.npy', tss_no_prior)
   

if __name__ == "__main__":                                                 
    parser=argparse.ArgumentParser(description='Optional Arguments')       
    parser.add_argument('--N', type = int, default=10000) 
    parser.add_argument('--label', type = int, default=1)         
    parser.add_argument('--threshold', type = int, default=1000)
    parser.add_argument('--cut_n_sigma', type = int, default=5)
    parser.add_argument('--cpus', type = int, default=10)
    parser.add_argument('--ts_no_prior',type = bool, default = False)
    args= parser.parse_args()                                              
    do_background_trials(args.N, args.label, args.threshold, args.cut_n_sigma, args.cpus, args.ts_no_prior)
 
