#!/usr/bin/env python                                                                
                                                                                     
import sys, getopt                                                                   
from icecube.tableio import I3TableWriter                                            
import numpy as np                                                                   
from I3Tray import *                                                                 
                                                                                     
import sys                                                                           
import os                                                                            
from icecube import icetray, dataio, dataclasses                                     
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/data/user/xk35/software/external/i3deepice/'))                                                 
from i3deepice.i3module import DeepLearningModule, print_info                        
                                                                                    
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/data/user/xk35/docker2/zstd/')) 
#import zstd
 
#from icecube.stochasticity_calculator import Calculator                             
import argparse, glob                                                                

'''
def cut_BDT(frame):
    BDT_score = frame['BDT_score'].value
    return BDT_score >= 0.025
'''

def DNNClassifier(frame):
    DNN_Classifier = [frame['TUM_dnn_classifiation_test_base']['Cascade'], 
                  frame['TUM_dnn_classifiation_test_base']['Skimming'], 
                  frame['TUM_dnn_classifiation_test_base']['Starting_Track'], 
                  frame['TUM_dnn_classifiation_test_base']['Stopping_Track'], 
                  frame['TUM_dnn_classifiation_test_base']['Through_Going_Track']]

    
    if np.argmax(DNN_Classifier)== 2:
        frame['DNN_Classifier_Starting_Track'] = dataclasses.I3Double(1.0)
    else:
        frame['DNN_Classifier_Starting_Track'] = dataclasses.I3Double(0.0)

    return True


def BDT2_cut(frame):
    BDT2_cut_score = 0.4
    logE_cut = 4.5
    
    BDT2_score = frame['BDT2_score'].value
    logE = np.log10(frame['SplineMPEMuEXDifferential'].energy)
    
    if frame['DNN_Classifier_Starting_Track'].value == 1.0:
        return True

    else: 
        return BDT2_score > BDT2_cut_score or logE < logE_cut

def main(argv):                                                                      
                                                                                     
                                                                                     
                                                                                     
    try:                                                                             
                                                                                     
        opts, args = getopt.getopt(argv,"hi:g:o:",["ifile=","gcd=","ofile="])         
                                                                                     
    except getopt.GetoptError:                                                       
                                                                                     
        print('test.py -i <inputfile> -g <gcd file>, -o <outputfile>')               
                                                                                     
        sys.exit(2)                                                                  
                                                                                     
    for opt, arg in opts:                                                            
                                                                                     
        if opt == '-h':                                                              
                                                                                     
            print('test.py -i <input file name> -g <gcd file> -o <output file directory>')                                                                                
                                                                                     
            sys.exit()                                                               
                                                                                     
        elif opt in  ("-i", "--ifile"):                                              
                                                                                     
            inputfile = arg                                                          
                                                                                     
        elif opt in ("-g", "--gcd"):                                                 
                                                                                     
            gcdfile = arg                                                            
                                                                                     
        elif opt in ("-o", "--ofile"):                                               
                                                                                     
            outputfile = arg                                                         

    print(gcdfile)
    output_name = inputfile.split('/')[-1]                                         
    output_name = output_name.replace('Level4', 'Level5') #('postBDTI','theo')                            

    outfile = (outputfile + '/' + output_name)                                     
    save_as_base = 'TUM_dnn_classifiation_test'                                    
    save_as = [save_as_base + '_base', save_as_base + '_rm_sat_wnd']               
                                                                                   
                                                                                   
                                                                                   
    tray = I3Tray()                                                                
    files = [gcdfile,inputfile]                                                    
                                                                                   
    tray.AddModule('I3Reader', 'reader', Filenamelist=files)    

    #tray.AddModule(cut_BDT, 'BDTcut')                   
    
    tray.AddModule(DeepLearningModule, 'DeepLearningMod',                            
                batch_size=640,                                                      
                cpu_cores=1,                                                         
                gpu_cores=1,                                                         
                model='classification',                                              
                add_truth=False, ## If true, classification truth from I3MCTree is added to the frame                                                                     
                pulsemap='InIceDSTPulses',                                           
                save_as= 'TUM_dnn_classifiation_test_base')                          

    tray.AddModule(DNNClassifier, 'EventTypeDNNClassifier')
    tray.AddModule(BDT2_cut, 'BDT2cut')



    tray.AddModule('I3Writer', 'writerI3',                                
                   Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], 
                   DropOrphanStreams=[icetray.I3Frame.DAQ],               
                   FileName=outfile)                                      



    tray.AddModule('TrashCan', 'YesWeCan')   
    tray.Execute()                           
                                             
if __name__=="__main__":                     
    main(sys.argv[1:])                       

