#!/usr/bin/env python

import numpy as np
from icecube import astro
import csky as cy
import argparse
import getpass

username = getpass.getuser()

ana_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/GC')
ana = cy.get_analysis(cy.selections.repo, cy.selections.GCDataSpecs.GC_2011_2022, dir = ana_dir, min_sigma = 0)


for a in ana.anas:
    a.bg_space_param = cy.pdf.BgAzimuthSinDecParameterization(a.bg_data, smooth=(1,4), hkw=a.kw_space_bg['hkw'])

### Discovery Potential ###
source_ra, source_dec = 4.64964433, -0.50503147
src = cy.sources(source_ra, source_dec)


def find_n_sig(bg, sig, dt, gamma, threshold, n_cut_sigma, t0, nsigma=None):                                     
    inj_conf = {                                                               
    'src' : src,                                                               
    'flux' : cy.hyp.PowerLawFlux(gamma),                                       
    'box' : True,                                                              
    'sig' : 'tw',                                                              
    't0' : t0,                                                          
    'dt' : dt                                                                  
    }     
    
    conf_gauss = {
    'time': 'utf',
    'seeder': cy.seeding.GaussianUTFSeeder(threshold=threshold),
    'fitter_args': dict(_log_params='dt', _fmin_method='Minuit'),
    'concat_evs': True,
    'cut_n_sigma': n_cut_sigma,
    'update_bg': True,
    'extra_keep': ['energy'],
    'rates_by' : 'livetime'
}
                                                                                                                   
                                                                  
    tr = cy.get_trial_runner(                                                  
        src=src,                                                               
        conf = conf_gauss,                                                     
        inj_conf=inj_conf,                                                     
        ana=ana,                                                               
        dt_max = ana[0].livetime/86400.,                                       
        use_grl=True,                                                          
        mp_cpus=10,                                                            
        TRUTH = False                                                          
        )                                                                      
    # determine ts threshold                                                   
    if nsigma is not None:                                                     
        ts = bg.isf_nsigma(nsigma)
        beta = 0.5
    else:                                                                      
        ts = bg.median() 
        beta = 0.9
                                                                            
    # include background trials in calculation                                 
    trials = {0: bg.trials}                                                     
    trials.update(sig[dt])                                                     
                                                                                                
    result = tr.find_n_sig(ts, beta,  logging=True,tol=.05, trials=trials) 
    print(tr.to_E2dNdE(result['n_sig'],unit = 1e3, E0=100))
    print(result['n_sig'])
    return tr.to_E2dNdE(result['n_sig'],unit = 1e3, E0=100), result['n_sig']


def calc(gamma, nsigma,threshold, n_cut_sigma, t0):
    trials_dir = ('/data/ana/analyses/NuSources/2023_galactic_center_analysis')
    # sig trial
    time_windows = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10.0, 100.0, 1000.0]
    sig = {}
    for dt in time_windows:
        #print(dt)
        sig_dir = (f'{trials_dir}/sig_trials/gamma/{gamma}/dt/{dt}/n_sig/')
        sig[dt] = cy.bk.get_all(sig_dir,
                                'trial*npy',
                                merge=np.concatenate,
                                post_convert=cy.utils.Arrays, log=False)

    # bg trials
    def ndarray_to_Chi2TSD(trials):
        return cy.dists.Chi2TSD(cy.utils.Arrays(trials))


    bg_dir = f'{trials_dir}/bg_trials/'
    bg = cy.bk.get_all(
            bg_dir,
            'trial*.npy',
            merge=np.concatenate,
            post_convert=ndarray_to_Chi2TSD, log = False)



    res = []
    nsigs = []
    for dt in time_windows:
        r, nsig = find_n_sig(bg = bg, sig = sig, dt=dt, gamma = gamma, threshold = threshold, t0 = t0, n_cut_sigma = n_cut_sigma, nsigma = nsigma)
        res.append(r)
        nsigs.append(nsig)
    cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/results')
    np.save(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/results/res_nsigma{nsigma}_gamma_{gamma}.npy', res)
    np.save(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/results/nsig_nsigma{nsigma}_gamma_{gamma}.npy', nsigs)

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Optional Arguments')
    parser.add_argument('--gamma', type = float, default=2.7)
    parser.add_argument('--nsigma', type = int, default=3)
    parser.add_argument('--threshold', type = int, default=1000)
    parser.add_argument('--n_cut_sigma', type = int, default=5)
    parser.add_argument('--t0', type = int, default=57570)
    args= parser.parse_args()
    calc(args.gamma, args.nsigma, args.threshold, args.n_cut_sigma, args.t0)
