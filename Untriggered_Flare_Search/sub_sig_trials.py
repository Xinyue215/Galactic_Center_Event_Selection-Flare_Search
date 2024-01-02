#!/usr/bin/env python

### submission script for sig_trial.py.###

import numpy as np
import os, time
from csky.utils import ensure_dir
import argparse
from submitter import Submitter
import getpass

username = getpass.getuser()

def submit_jobs(N, njob, gamma, t0, threshold, cut_n_sigma, cpus):

	T = time.time()

	job_base_dir = ensure_dir('/scratch/{}/GC_Untriggered_Flare_Search/sig_trial_gamma{}_{:17.6f}'.format(username, gamma,T))

	sub = Submitter (job_dir=job_base_dir, memory=8, max_jobs=100, config = 'config_GC')

	

	commands = []
	labels = []
	sig_trials = os.path.abspath('sig_trials.py')

	time_windows = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2,1e3]

	gamma = gamma
	for dt in time_windows:
		if dt <= 1:                   
			n_sigs = np.r_[2:59:8]    
		elif dt in [1e1,1e2]:                  
			n_sigs = np.r_[20:400:15]  
		elif dt == 1e3:
			n_sigs = np.r_[100:801:50]


		for nsig in n_sigs:
			for l in range(njob):
				fmt = "{} --dt {} --nsig {} --gamma {} --label {} --N {} --t0 {} --threshold {} --cut_n_sigma {} --cpus {}"
				command=fmt.format(sig_trials, dt, nsig, gamma, l, N, t0, threshold, cut_n_sigma, cpus)
        
				label='runTrial_dt{}_nsig{}_gamma{}_label{}'.format(dt, nsig, gamma, l)
				
				labels.append(label)
				commands.append(command)
        
	commands = np.unique(commands)  
	labels = np.unique(labels)      
	sub.submit_npx4(commands,labels)
	

if __name__ == "__main__":
	parser=argparse.ArgumentParser(description='Optional Arguments')
	parser.add_argument('--N', type = int, default=1)
	parser.add_argument('--njob', type = int, default=1)
	parser.add_argument('--gamma', type = float, default=2.0)
	parser.add_argument('--t0', type = int, default = 57570)
	parser.add_argument('--threshold', type = int, default=1000)
	parser.add_argument('--cut_n_sigma', type = int, default=5)
	parser.add_argument('--cpus', type = int, default=10)
	args= parser.parse_args()
	submit_jobs(args.N, args.njob, args.gamma, args.t0, args.threshold, args.cut_n_sigma, args.cpus)



