#!/usr/bin/env python

import numpy as np
from icecube import astro
import csky as cy
from astropy.time import Time
import argparse
import getpass                                                                    
                                                                                  
username = getpass.getuser()                                                      
                                                                                  
ana_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/GC')                                                                 
ana = cy.get_analysis(cy.selections.repo, cy.selections.GCDataSpecs.GC_2011_2022, dir = ana_dir, min_sigma = 0)                                                     


for a in ana.anas:
    a.bg_space_param = cy.pdf.BgAzimuthSinDecParameterization(a.bg_data, smooth=(1,4), hkw=a.kw_space_bg['hkw'])


source_ra, source_dec = astro.gal_to_equa(0.,0.)
src = cy.sources(source_ra, source_dec)


# bg trials
trials_dir = ('/data/ana/analyses/NuSources/2023_galactic_center_analysis')
def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))


bg_dir = f'{trials_dir}/bg_trials/'
bg = cy.bk.get_all(
        bg_dir,
        'trial*.npy',
        merge=np.concatenate,
        post_convert=ndarray_to_Chi2TSD, log = False)

def do_background_trials(threshold, cut_n_sigma, cpus, unblind):
    conf_gauss = {
        'time': 'utf',
        #'box':True,
        'seeder': cy.seeding.GaussianUTFSeeder(threshold=threshold),
        'fitter_args': dict(_log_params='dt', _fmin_method='Minuit'),
        'concat_evs': True,
        'cut_n_sigma': cut_n_sigma,
        'update_bg': True,
        'extra_keep': ['energy'],
        'rates_by' : 'livetime'
    }


    # get trial runner
    tr = cy.get_trial_runner(src=src, conf = conf_gauss, ana=ana, mp_cpus=cpus, dt_max =  (ana[0].livetime/86400)/2, #TRUTH = unblind 
        use_grl = True)
    # unblind trial
    #unblind_trial = tr.get_many_fits(1, logging=True)
    unblind_trial = tr.get_one_fit(TRUTH = unblind)
    out_dir = cy.utils.ensure_dir(f'/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/trials/bg_trials/unblind_trial')
    filename = out_dir + '/unblinded_trial.npy'
    print('->', filename)
    # notice: trials.as_array is a numpy structured array, not a cy.utils.Arrays
    np.save(filename, unblind_trial)
    return tr, unblind_trial


def unblind(threshold, cut_n_sigma, cpus, unblind):
    tr, unblind_trial = do_background_trials(threshold, cut_n_sigma, cpus, unblind)
    p_val_nsigma = bg.sf_nsigma(unblind_trial[0], fit=True)
    p_val = bg.sf(unblind_trial[0], fit=True)
    
    print('***** P Value:', p_val, '*****')
    print('***** P Value n sigma:', p_val_nsigma, '*****')
#    print('ts',unblind_trial['ts'])
#    print('ns',unblind_trial['ns'])
#    print('gamma', unblind_trial['gamma'])
#    print('dt', unblind_trial['dt'])
#    print('t0', unblind_trial['t0'])
    print(tr.format_result(unblind_trial))    

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Optional Arguments')
    parser.add_argument('--threshold', type = int, default=1000)
    parser.add_argument('--cut_n_sigma', type = int, default=5)
    parser.add_argument('--cpus', type = int, default=10)
    parser.add_argument('--unblind', type = bool, default=False)
    args= parser.parse_args()
    unblind(args.threshold, args.cut_n_sigma, args.cpus, args.unblind)

