
# Galactic Center Analysis 

This project includes two parts:  
1. Event selection from L3 using  IC86 2011 - 2022 data
2. Untriggered flare search at the Galactic Center

Analysis wiki: https://wiki.icecube.wisc.edu/index.php/IC86_Galactic_Center_Point_Source

## Requirements:
- cvmfs */cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh*
- csky tag v1.1.14 *https://github.com/icecube/csky.git*
- Submitter *https://github.com/ssclafani949/Submitter*
- DeepIceLearning IceTray Module *https://github.com/tglauch/i3deepice*

To run scripts, load:
*eval /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh*
*/data/user/xk35/software/meta_projects/v01-01-00/build/env-shell.sh*

## Data processing:
Data processing takes 2 steps: 
1. L3 to L4
`process_to_level4.py` takes L3 i3 files and process to L4. It does **precuts**, apply **BDT classifier** and make cut based on BDT score,  calculate **stochasticity** and use the stochasticity as inputs for the **BDT regressor**. 

Run `process_to_level4.py` with 4 arguements:
- --ifile (input file)
- --gcd (gcd file)
- --ofile (output file)
- --cut_score (BDT classifier cut score: **0.025** for this dataset)

`sub_provess_level4.py` is the submission script for `process_to_level4.py`. It uses config as the config file. It submits jobs monthly. 

Submit `process_to_level4.py` to NPX by running `sub_provess_level4.py` with 3 arguements:

- --year (submit year from */data/ana/Muon/level3/exp/*)
- --month(submit month)
- --cut_score (BDT classifier cut score, default *0.025*)

2. L4 to L5
`theo_classifier.py` takes L4 i3 files and process to L5. It uses the  DeepIceLearning IceTray Module to do **event type classification**. A cut based on the output of the event type classification, energy, and BDT regressor score (from last step) is made. This is the **regressor cut** to cut muon bundles.

Run `theo_classifier.py` with 3 arguements:
- --ifile (input file)  
- --gcd (gcd file)      
- --ofile (output file) 
This script requires singularity environment. Run the script with *bash icetray_env_gpu.sh theo_classifier.py [arguments]*

Submit `theo_classifier.py` to NPX by running `sub_theo_Data.py` with 2 arguements:
- --year (from the directory created in step 1)
- --month(submit month) 


When have the final level i3 files, concatenate the i3 files and convert to hdf5. The script `save_to_npy.py` converts hdf5 files to numpy.

The notebook `Check_Bad_runs.ipynb` loads the numpy data files and check if there exists bad runs.

## Untriggered flare search:
The signal and background trials run with `sig_trials.py` and `bg_trials.py` respectively. 

Run `bg_trials.py`with 5 arguements:
- --N (number of trials)
- --label (label the trial in output file)
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --cpus (number of cpus used, default 10)

The output numpy files are saved at `/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/trials/bg_trials`.


Run `sig_trials.py` with 7 arguements:
- --dt (time window)
- --nsig (number of injected signals)
- --gamma (spectral index)
- --label (label the trial in output file)
- --N (number of trials)
- --t0 (mean time of flare, default 57570)
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --cpus (number of cpus used, default 10)

The output numpy files are saved at `/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/trials/sig_trials`.


**Before submitting to NPX, please make sure** `config_GC` **is located in $HOME, Submitter finds the submit config file in a path relative to $HOME**
The submission script for the background trials `sub_bg_trials.py` runs with config `config_GC`. It takes 5 arguements:
- --N (number of trials)
- --njob (number of jobs)
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --cpus (number of cpus used, default 10)

It submits **njobs** and each job is running **N** trials. The script creates a job base directory in `/scratch/{username}/GC_Untriggered_Flare_Search/`.
To run `sub_bg_trials.py` in NPX, use command: `python sub_bg_trials.py --N {} --njob {} --threshold {} --cut_n_sigma {} --cpus {}`. To make `bg_trials.py` excutable, use command `chmod +x bg_trials.py`.

The submission script for the signal trials `sub_sig_trials.py` runs with config `config_GC`. It takes 7 arguements:
- --N (number of trials) 
- --njob (number of jobs)
- --gamma (spectral index)
- --t0 (mean time of flare, default 57570)
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --cpus (number of cpus used, default 10)


It submits **njobs** with spectral index **gamma** for time windows **1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3**. Each job is running **N** trials. The script creates a job base directory in `/scratch/{username}/GC_Untriggered_Flare_Search/`.
To run `sub_sig_trials.py` in NPX, use command: `python sub_sig_trials.py --N {} --njob {} --gamma {} --t0 {} --threshold {} --cut_n_sigma {} --cpus {}`. To make `sig_trials.py` excutable, use command `chmod +x bg_trials.py`.


The notebook `bg_trials.ipynb` plots the 100,000 bg trials with the best fit gamma, dt, T_0, and n_sig. Recreates plot *https://wiki.icecube.wisc.edu/index.php/File:Bg_trials_.png*. 

The notebook `bias_check.ipynb` plots bias with signal trials. Recreates plots *https://wiki.icecube.wisc.edu/index.php/IC86_Galactic_Center_Point_Source/Untriggered_Flare_Search#Bias_Check*

The script `discovery_potential.py` calculates discovery potential using the signal and background trials. Can also be used to calculate sensitivity.

Run `discovery_potential.py` with 2 arguements:
- -- gamma (spectral index)
- -- nsigma (n sigma for discovery potential. If want to calculate sensitivity, pass --nsigma None)
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --t0 (mean time of flare, default 57570)
The output will be saved to direstory `/data/user/{username}/Galactic_Center/Untriggered_Flare_Search/results`. 

### Unblinding
Run script `unblind.py` to unblind. It takes 4 arguements:
- --threshold (threshold of S/B, default 1000)
- --cut_n_sigma (n sigma cut in gaussian config, default 5)
- --cpus (number of cpus used, default 10)
- --unblind (boolean, default False. Pass True to unblind)
