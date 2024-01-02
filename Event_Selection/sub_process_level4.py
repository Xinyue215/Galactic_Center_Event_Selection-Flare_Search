#!/usr/bin/env python


#from submitter import Submitter
import numpy as np
import glob
import os, time
from csky.utils import ensure_dir
import argparse
from submitter import Submitter
import shutil

def submit_jobs(year, month, cut_score):
    

    blacklist = ['c9-5.icecube.wisc.edu']
                                                                                    
    input_base_dir = f'/data/ana/Muon/level3/exp/{year}/'+str(month)+'*'
                      
    filenames = sorted(glob.glob(input_base_dir + '/*/Level3*.i3.zst')) 

    T = time.time()                                                                                       
    job_base_dir = ensure_dir('/scratch/xk35/Event_Selection/level4/Data/{}/Data_{}_{:17.6f}'.format(year,str(month),T))                                     
    sub = Submitter (job_dir=job_base_dir, memory=8, max_jobs=1000, config = 'config')                                       
    env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'


    commands = []                                                                                         
    labels = []                                                                                           
    print('number of files about to process:'+str(len(filenames)))
    for i,filename in enumerate(sorted(filenames)):                                                                  
        date = filename.split('/')[-3]
        output_base_dir = ensure_dir(f'/data/ana/PointSource/Galactic_Center/i3_processing/L4/exp/IC86_{year}/'+ str(date)+'/')

        folder_group = (filename.split('/')[-1]).split('_')[-3]

        ensure_dir(output_base_dir + str(folder_group))
        base_dir = f'/data/ana/Muon/level3/exp/{year}/'+str(date)
        gcd_file = glob.glob(base_dir + '/' + folder_group +'/*GCD*')[0]


        fmt = "{} /home/xk35/BDT_corrected/From_I3_File/Final/process_to_level4.py -i {} -g {} -o {} -s {}" 
        command = fmt.format(env_shell, filename, gcd_file, output_base_dir+str(folder_group), cut_score)                 

        label = 'processFile_{}'.format(filename.split('/')[-1])                                                           

        commands.append(command)                                                                          
        labels.append(label)                                                                              

    #print(labels)    

    commands = np.unique(commands)
    labels = np.unique(labels)
    sub.submit_npx4(commands,labels,blacklist = blacklist)

    gcd_file = glob.glob(input_base_dir + '/*/Level2*.i3*')
    for gcd in gcd_file:
        try:
            shutil.copyfile(gcd, f'/data/ana/PointSource/Galactic_Center/i3_processing/L4/exp/IC86_{year}/'+gcd.split('/')[-3] + '/' + gcd.split('/')[-2] + '/' + gcd.split('/')[-1])
        except:
            continue

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Optional Arguments')
    parser.add_argument('--year', type = str, default='2011')
    parser.add_argument('--month', type = str, default='01')
    parser.add_argument('--cut_score', type = float, default=0.025)
    args= parser.parse_args()
    submit_jobs(args.year, args.month,  args.cut_score)
