#!/usr/bin/env python

##################################################################################
#### This scriprt includes: precuts, first BDT cut, stochastivity, second BDT ####
##################################################################################

import sys, getopt                                                                
from icecube import dataio                                                        
import numpy as np                                                                
import glob                                                                       
from icecube import icetray, dataio, tableio,dataclasses,simclasses,recclasses, gulliver, finiteReco, paraboloid,phys_services, lilliput, gulliver_modules, rootwriter, common_variables, spline_reco
from I3Tray import *                                                              
from icecube.hdfwriter import I3SimHDFWriter                                      
from icecube import millipede, linefit                                            
from icecube.hdfwriter import I3HDFWriter                                         
from icecube import astro,icetray                                                 
import time,pickle                                                                
import sklearn, xgboost
from icecube.stochasticity_calculator import Calculator                            
import subprocess                                                                 
from csky.utils import ensure_dir                                                 
import gc                                                                         
import joblib                                                                     
from icecube import hdfwriter,bayesian_priors                                      


########### PRECUTS############

GC_ra , GC_dec = astro.gal_to_equa(0., 0.)
def zenith_cut(frame, minDec=GC_dec-np.radians(10), maxDec = GC_dec+np.radians(10)):
    dec = frame['SplineMPE'].dir.zenith - np.radians(90.)

    return ((dec >= minDec) and (dec <= maxDec))

def splineMPE_cut(frame):
    status = frame['SplineMPE'].fit_status
    return  (status == 0)

def rlogl_cut(frame):
    rlogl = frame['SplineMPEFitParams'].rlogl
    return rlogl<9.

def ldirE_cut(frame):
    
    ldirE = frame['SplineMPEDirectHitsE'].dir_track_length


    return ldirE >250.

def sigma_cut(frame):
    print('doing sigma cut')
    err1 = frame['MPEFitParaboloidFitParams'].pbfErr1
    err2 = frame['MPEFitParaboloidFitParams'].pbfErr2
    sigma = np.sqrt((err1**2+err2**2)/2)
    return sigma <np.radians(4.5)
    sys.stdout.flush()

def delta_angle(pt1_zen, pt1_azi=None, pt2_zen=None, pt2_azi=None):


    z1=np.array(pt1_zen)
    z2=np.array(pt2_zen)
    a1=np.array(pt1_azi)
    a2=np.array(pt2_azi)
    pi = np.pi
    # haversine!
    cos_alpha = np.cos(z1-z2) - np.cos(pi/2.0 - z1)*np.cos(pi/2.0 - z2)*(1-np.cos(a1-a2))
    alpha=180/pi*np.arccos(cos_alpha)
    if np.isnan(alpha):
        alpha = 180
    return alpha

def SplitMinZenithFunc(zen1,zen2,zen3,zen4):
    zen1[(zen1<0)|(zen1!=zen1)]=400
    zen2[(zen2<0)|(zen2!=zen2)]=400
    zen3[(zen3<0)|(zen3!=zen3)]=400
    zen4[(zen4<0)|(zen4!=zen4)]=400
    return np.degrees(np.min(np.transpose((zen1,zen2,zen3,zen4)),axis=1))


#BayesianFunc = lambda blogl,logl: np.nan_to_num(blogl - logl)


################# BDT I CUT ##################

def BDT_score(frame,cut_score):
    ndirE = frame['SplineMPEDirectHitsE'].n_dir_pulses
    rlogl = frame['SplineMPEFitParams'].rlogl
    ##
    err1 = frame['MPEFitParaboloidFitParams'].pbfErr1
    err2 = frame['MPEFitParaboloidFitParams'].pbfErr2
    sigma = np.sqrt((err1**2+err2**2)/2)
    ##
    highNoise_zen = frame['MPEFitHighNoise'].dir.zenith
    highNoise_azi = frame['MPEFitHighNoise'].dir.azimuth
    TWHV_zen = frame['MPEFit_TWHV'].dir.zenith
    TWHV_azi = frame['MPEFit_TWHV'].dir.azimuth
    MPEHighNoise_delta_angle = delta_angle(highNoise_zen,highNoise_azi,TWHV_zen,TWHV_azi)
    logE = np.log10(frame['SplineMPEMuEXDifferential'].energy)

#    Bayesian_logl = frame['SPEFit2BayesianFitParams'].logl
#    TWHV_logl = frame['SPEFit2_TWHVFitParams'].logl
    #BayesLLHRatio = BayesianFunc(Bayesian_logl,TWHV_logl)
    LLH = frame['SPEFit2_TWHVFitParams'].logl

    cog_z = frame['HitStatisticsValues'].cog.z
    cog_r2 = frame['HitStatisticsValues'].cog.x**2 + frame['HitStatisticsValues'].cog.y**2

    ldirE = frame['SplineMPEDirectHitsE'].dir_track_length
    spMPE_zen = frame['SplineMPE'].dir.zenith
    spMPE_azi = frame['SplineMPE'].dir.azimuth
    LineFit_TWHV_zen = frame['LineFit_TWHV'].dir.zenith
    LineFit_TWHV_azi = frame['LineFit_TWHV'].dir.azimuth
    LineFit_delta_angle = delta_angle(spMPE_zen,spMPE_azi,LineFit_TWHV_zen,LineFit_TWHV_azi)
    nearlyE = frame['SplineMPEDirectHitsE'].n_early_pulses
    NEarlyNCh = nearlyE/ndirE
    MuExrllt = frame['MuEXAngular4_rllt'].value

    feature = np.zeros((1,12))
    feature[0][:] = (np.array([ndirE, rlogl, sigma,MPEHighNoise_delta_angle,logE,LLH,cog_r2,cog_z,ldirE,LineFit_delta_angle,NEarlyNCh,MuExrllt]))
    print(feature)
    clf = xgboost.XGBClassifier()
    clf.load_model('/home/xk35/BDT_corrected/Train/BDT_I_GC_xgb1.4.json')
    score = clf.predict_proba(feature)[0, 1]
    print(float(score))
    sys.stdout.flush()
    frame['BDT_score'] = dataclasses.I3Double(float(score))

    return score >= float(cut_score)
    #return True

########### Stochasticity #############

           
load( "libtruncated_energy" )               
@icetray.traysegment

def Truncated( tray, Name, Pulses = "", Seed = "", Suffix = "",
    If = lambda f: True, PhotonicsService = "", Model = "" ):
    # Base name to put into frame
    
    print('starting truncated reco')
    TruncatedName = Seed + "TruncatedEnergy" + Suffix + Model

    tray.AddModule( "I3TruncatedEnergy",
        #Name of pulses to grab from frame.
        RecoPulsesName          = Pulses,
        #Name of the reconstructed particle to use.
        RecoParticleName        = Seed,
        #Name of the result particle to put in frame.
        ResultParticleName      = TruncatedName,
        #Photonics service to use for energy estimator.
        I3PhotonicsServiceName  = PhotonicsService,
        #Calibration info for Relative DOM Efficiency (RDE).  For HQE DOMs.
        UseRDE                  = True,
        If                      = If )
                
def successfulreco_cut(frame):
    print('starting BINS cut')
        
    return  frame.Has('SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon')
                                               

################## BDT II ###################
def BDT2_score(frame):
    PeakOverMean = frame['Stochasticity'].get('PeakOverMean')
    PeakOverMedian = frame['Stochasticity'].get('PeakOverMedian')
    chi2 = frame['Stochasticity'].get('chi2')
    chi2_new = frame['Stochasticity'].get('chi2_new')
    chi2_red = frame['Stochasticity'].get('chi2_red')
    chi2_red_new = frame['Stochasticity'].get('chi2_red_new')
    combined = frame['Stochasticity'].get('combined')



    feature = np.zeros((1,7))
    feature[0][:] = (np.array([PeakOverMean, PeakOverMedian, chi2, chi2_new, chi2_red,chi2_red_new, combined]))
    #print(feature)
    #clf = joblib.load('/home/xk35/BDT_corrected/Train/xgb_Train_nobayes.joblib')    
    clf = xgboost.XGBRegressor()
    clf.load_model('/home/xk35/BDT_corrected/Train/BDT_v2.json')

    score = clf.predict(feature)
    sys.stdout.flush()
    frame['BDT2_score'] = dataclasses.I3Double(float(score))
    print(score)
    #return score >= float(cut_score)
    return True





def main(argv):
    inputfile = ''
    outputfile = ''
    folder_group = ''
    try:
        opts, args = getopt.getopt(argv,"hi:g:o:s:",["ifile=","gcdfile","ofile=","cut_score="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -g <gcdfile> -o <outputfile> -s <cutscore>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <input file name> -g <gcd file name> -o <output file directory> <True/False> -s <cut score>')
            sys.exit()
        elif opt in  ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-g", "--gcd"):
            gcdfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-s", "--cut_score"):
            cut_score = arg


    print('Input file is "', inputfile)
    print('Output file is "', outputfile)


    tray = I3Tray()
    files = [gcdfile,inputfile]
    output_name = inputfile.split('/')[-1]
    output_name = output_name.replace('Level3','Level4')
    outfile = (outputfile + '/' + output_name)
    tray.AddModule('I3Reader', 'reader', Filenamelist=files)

    # Precut
    tray.AddModule(zenith_cut,     
                   "MPEFitZenithFilter",     
                   minDec=GC_dec-np.radians(10), maxDec = GC_dec+np.radians(10))
    tray.AddModule(splineMPE_cut,
                    "MPEFitFilter")
    tray.AddModule(rlogl_cut,
                "MPEFitrloglFilter")
    
    try:
        DirectHitsDefs=common_variables.direct_hits.default_definitions
    except BaseException:
        print('Base Exception')
        DirectHitsDefs=common_variables.direct_hits.get_default_definitions()
    
    DirectHitsDefs.append(common_variables.direct_hits.I3DirectHitsDefinition("E",-15.,250.))
    print('DirectHitsDefs', DirectHitsDefs)
    for track in ["SplineMPE"]:
        print('track++++++++++++++++++++++++++', track)
        tray.Add("Delete", "cleanUpDirectHits",
                 KeyStarts = ['SplineMPEDirectHits'])
        tray.AddSegment(common_variables.direct_hits.I3DirectHitsCalculatorSegment,
            DirectHitsDefinitionSeries=DirectHitsDefs,
            PulseSeriesMapName               = "SRTHVInIcePulses",#"TWSRTHVInIcePulsesIC",
            ParticleName                     = track,
            OutputI3DirectHitsValuesBaseName = track+'DirectHits',
            BookIt                           = True)



    tray.AddModule(ldirE_cut,
                    "MPEFitldirEFilter")
    tray.AddModule(sigma_cut,
                    "SigmaFilter")

    # BDT
    tray.AddModule(BDT_score,"BDT_cut", cut_score = cut_score)

    # Stoch
    old_list = ['SplineMPETruncatedEnergy_SPICEMie_BINS_dEdxVector', 'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Muon', 'SplineMPETruncatedEnergy_SPICEMie_AllBINS_dEdX', 'SplineMPETruncatedEnergy_SPICEMie_DOMS_dEdX', 'SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon', 'SplineMPETruncatedEnergy_SPICEMie_BINS_dEdX', 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_MuEres', 'SplineMPETruncatedEnergy_SPICEMie_BINS_MuEres', 'SplineMPETruncatedEnergy_SPICEMie_ORIG_dEdX', 'SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino', 'SplineMPETruncatedEnergy_SPICEMie_AllBINS_MuEres', 'SplineMPETruncatedEnergy_SPICEMie_AllBINS_Neutrino', 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon', 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Neutrino', 'SplineMPETruncatedEnergy_SPICEMie_AllDOMS_dEdX', 'SplineMPETruncatedEnergy_SPICEMie_BINS_Muon', 'SplineMPETruncatedEnergy_SPICEMie_BINS_Neutrino', 'SplineMPETruncatedEnergy_SPICEMie_DOMS_MuEres', 'SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon', 'SplineMPETruncatedEnergy_SPICEMie_ORIG_Neutrino']

    for i in old_list:
         tray.AddModule('Rename', Keys=[i, 'old'+i])



    tray.AddService( "I3PhotonicsServiceFactory", "PhotonicsServiceMu_SpiceMie",
    PhotonicsTopLevelDirectory  ="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/",
    DriverFileDirectory         ="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/driverfiles",
    PhotonicsLevel2DriverFile   = "mu_photorec.list",
    PhotonicsTableSelection     = 2,
    ServiceName                 = "PhotonicsServiceMu_SpiceMie" )

    tray.AddSegment( Truncated, Pulses = "SRTHVInIcePulses", Seed = "SplineMPE",
    Suffix = "", PhotonicsService = "PhotonicsServiceMu_SpiceMie",
    Model = "_SPICEMie" )

    tray.AddModule(Calculator.Stochasticity,'Stochasticity',
                        UseTruncatedInputs   = False,
                        Truncated_Seed       = "SPlineMPE",
                        Truncated_Suffix     = "",
                        Truncated_Model      = "_SPICEMie",
                        Truncated_dEdxVector = "SplineMPETruncatedEnergy_SPICEMie_BINS_dEdxVector",
                        OutputName           = "Stochasticity",
                        )
    #print('done sto calc')

    #tray.AddModule('I3Writer', 'writerI3',
    #               Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics],
    #               DropOrphanStreams=[icetray.I3Frame.DAQ],
    #               FileName=outfile)
    #print('dropped frames')

    # BDT II
    tray.AddModule(BDT2_score,"BDT2")

    tray.Add('I3Writer','EventWriter',
        Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics],
        DropOrphanStreams=[icetray.I3Frame.DAQ],
        Filename=outfile)
    tray.AddModule('TrashCan','can')
    tray.Execute()
    tray.Finish()

if __name__ == "__main__":
    main(sys.argv[1:])
































