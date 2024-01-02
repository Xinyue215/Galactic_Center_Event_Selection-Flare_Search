#!/usr/bin/env python


#from submitter import Submitter   
import numpy as np                 
import glob                        
import os, time                    
from csky.utils import ensure_dir  
import argparse                    
from submitter import Submitter    



def submit_jobs(year, month):

    input_base_dir = f'/data/ana/PointSource/Galactic_Center/i3_processing/L4/exp/IC86_{year}/'+str(month)+'*'
                 
    filenames = sorted(glob.glob(input_base_dir + '/*/Level4*.i3.zst'))

    T = time.time()                                                                                       
    job_base_dir = ensure_dir('/scratch/xk35/Event_Selection/level5/Data/{}/{}_T_{:17.6f}'.format(year, str(month),T))                                     

    sub = Submitter (job_dir=job_base_dir, memory=8, max_jobs=3000, config = config_theo)  
    env_shell = 'icetray_env_gpu.sh'                                        


    commands = []                                                                                         
    labels = []                                                                                           
    print('number of files about to process:'+str(len(filenames)))
    for i,filename in enumerate(sorted(filenames)):
        date = filename.split('/')[-3]
        output_base_dir = ensure_dir(f'/data/ana/PointSource/Galactic_Center/i3_processing/L5/exp/IC86_{year}/' + str(date)+'/')

        folder_group = filename.split('/')[-2]
        ensure_dir(output_base_dir + str(folder_group))


        base_dir = f'/data/ana/PointSource/Galactic_Center/i3_processing/L4/exp/IC86_{year}/'+str(date)     
        gcd_file = glob.glob(base_dir + '/' + folder_group +'/*GCD*')[0]   
        fmt = "{} /home/xk35/BDT_corrected/From_I3_File/Final/theo_classifier.py -i {} -g {} -o {}" 
        command = fmt.format(env_shell, filename, gcd_file, output_base_dir + str(folder_group))                

        label = 'processFile_{}'.format(filename.split('/')[-1])                                                           

        commands.append(command)                                                                          
        labels.append(label)                                                                              

    blacklist = ['gtx-6.icecube.wisc.edu', 'gtx-30.icecube.wisc.edu','gtx-32.icecube.wisc.edu', 'gtx-5.icecube.wisc.edu']
    reqs = 'has_avx && CUDACapability'

    sub.submit_npx4(commands,labels, blacklist = blacklist, reqs = reqs, gpus = 1)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Optional Arguments')
    parser.add_argument('--year', type = str, default='2011')
    parser.add_argument('--month', type = str, default='01')
    args= parser.parse_args()
    submit_jobs(args.year, args.month)



