#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from icecube import astro, dataio,paraboloid
import histlite as hl
import csky as cy
from csky import hyp
import pandas as pd
#from importlib import reload
import tables
import argparse, glob
import sys, getopt


def main(year):


    GC_ra , GC_dec = astro.gal_to_equa(0., 0.)
    sindec_uplim = np.sin(GC_dec + np.radians(10))
    sindec_lolim = np.sin(GC_dec - np.radians(10))
    
    
    Data = tables.open_file(f'/data/ana/PointSource/Galactic_Center/i3_processing/L5/exp/IC86_{year}/Level5_IC86.{year}_data.hdf5')
    
    dtype = [('run', int), ('event', int), ('subevent', int),
                  ('ra', float), ('dec', float),
                    ('azi', float), ('zen', float), ('time', float),
                   ('logE', float), ('angErr', float)]
    
    data = np.zeros(len(Data.root.I3EventHeader.cols.Run[:]), dtype=dtype)
    data['run'] =Data.root.I3EventHeader.cols.Run[:]
    data['event'] = Data.root.I3EventHeader.cols.Event[:]
    data['subevent'] = Data.root.I3EventHeader.cols.SubEvent[:]
    data['azi'] = Data.root.SplineMPE.cols.azimuth[:]
    data['zen'] = Data.root.SplineMPE.cols.zenith[:]
    data['time'] = Data.root.I3EventHeader.cols.time_start_mjd[:]
    data['logE'] = np.log10(Data.root.SplineMPEMuEXDifferential.cols.energy[:])
    Data_err1 = Data.root.MPEFitParaboloidFitParams.cols.err1[:]
    Data_err2 = Data.root.MPEFitParaboloidFitParams.cols.err2[:]
    data['angErr'] = np.sqrt((Data_err1**2+Data_err2**2)/2)
    
    #data['ra'], data['dec'] = astro.dir_to_equa(data['zen'], data['azi'], data['time'])
    data['ra'] = data['azi']
    data['dec'] = data['zen']-np.pi/2
    
    np.save(f'/data/ana/PointSource/Galactic_Center/current/IC86_{year}_L5_data.npy', data)

if __name__ == "__main__":                                               
    parser=argparse.ArgumentParser(description='Optional Arguments')     
    parser.add_argument('--year', type = str, default='2017')   
    args= parser.parse_args()                                            
    main(args.year)                                                 
                                                                         

