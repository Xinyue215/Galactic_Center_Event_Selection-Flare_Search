#!/usr/bin/env python

#### submission script for bg_trias.py. This script submits njobs each job runs N trials. ###


import numpy as np
import os, time
from csky.utils import ensure_dir
import argparse
from submitter import Submitter
import getpass
username = getpass.getuser()


def submit_jobs(N, njob, threshold, cut_n_sigma, cpus, ts_no_prior):

	T = time.time()

	job_base_dir = ensure_dir(f'/scratch/{username}/GC_Untriggered_Flare_Search/bg_trials_neg_TS_{T:17.6f}')

	sub = Submitter (job_dir=job_base_dir, memory=8, max_jobs=100, config = 'config_GC')

	

	commands = []
	labels = []
	bg_trials = os.path.abspath('bg_trials.py')
	for i in range(njob):
		fmt = "{} --N {} --label {} --threshold {} --cut_n_sigma {} --cpus {} --ts_no_prior {}"
		commands.append(fmt.format(bg_trials, N, i, threshold, cut_n_sigma, cpus, ts_no_prior))
		labels.append('run_bg_Trial_{}'.format(i))

	commands = np.unique(commands)
	labels = np.unique(labels)
	sub.submit_npx4(commands,labels)
	

if __name__ == "__main__":
	parser=argparse.ArgumentParser(description='Optional Arguments')
	parser.add_argument('--N', type = int, default=1)       
	parser.add_argument('--njob', type = int, default=1)        
	parser.add_argument('--threshold', type = int, default=1000)
	parser.add_argument('--cut_n_sigma', type = int, default=5)
	parser.add_argument('--cpus', type = int, default=10)
	parser.add_argument('--ts_no_prior',type = bool, default = False) 
	args= parser.parse_args()
	submit_jobs(args.N, args.njob, args.threshold, args.cut_n_sigma, args.cpus, args.ts_no_prior)



