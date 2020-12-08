import pathlib
import cdflib
import os, glob
from spacepy import pycdf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import matplotlib.colors as col
import matplotlib.dates as mdates
from matplotlib import gridspec

from datetime import datetime,date,timedelta

##### 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [array[idx],idx]

def date_finder(event_point):
    event_time = datetime.strptime(event_point,"%Y-%m-%d/%H:%M:%S")
    SEA_day=[event_time - timedelta(days=before_event_day)+timedelta(days=t) for t in range(0,lasting_event_day)]
    return SEA_day, event_time

def append_orb(event_point):
    epoch_orb=np.empty((0,1)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pos_lstar_op=np.empty((0,9))
    pos_blocal_op=np.empty((0,1))
    pos_beq_op=np.empty((0,1))
    for i in date_finder(event_point)[0]:
        path=common_path+"ergsc/satellite/erg/orb/l3/opq/"+i.strftime('%Y')+"/"+i.strftime('%m')+"/"
        file=path+"erg_orb_l3_op_"+i.strftime('%Y%m%d')+"_v02.cdf"
        if pathlib.Path(file).exists():
            data=cdflib.CDF(file)
            epoch_orb=np.append(epoch_orb,data['epoch'])
            pos_lstar_op=np.vstack((pos_lstar_op,data['pos_lstar_op']))
            pos_blocal_op=np.append(pos_blocal_op,data['pos_blocal_op']) 
            pos_beq_op=np.append(pos_beq_op,data['pos_beq_op'])
        else: print("No file path:"+file)
    pos_lstar_op[pos_lstar_op<0]=0 #np.nan
    return epoch_orb,pos_lstar_op, pos_blocal_op, pos_beq_op

def erg_orb_l3(event_point):
    epoch_orb=np.empty((0,)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pos_lstar_op=np.empty((0,9))
    pos_blocal_op=np.empty((0,))
    pos_beq_op=np.empty((0,))
    for i in date_finder(event_point)[0]:
        path=common_path+"rgsc/satellite/erg/orb/l3/opq/"+i.strftime('%Y')+"/"+i.strftime('%m')+"/"
        file=path+"erg_orb_l3_op_"+i.strftime('%Y%m%d')+"_v02.cdf"
        if pathlib.Path(file).exists():
            data=cdflib.CDF(file)
            epoch_orb=np.concatenate((epoch_orb,data['epoch']))
            pos_lstar_op=np.concatenate((pos_lstar_op,data['pos_lstar_op']))
            pos_blocal_op=np.concatenate((pos_blocal_op,data['pos_blocal_op']))
            pos_beq_op=np.concatenate((pos_beq_op,data['pos_beq_op']))
        else: print("No file path:"+file)
    pos_lstar_op[pos_lstar_op<0]=0 #np.nan
    return epoch_orb,pos_lstar_op, pos_blocal_op, pos_beq_op

def erg_orb_l3_t89(event_point):
    epoch               = np.empty((0,)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pos_eq_t89          = np.empty((0,2))
    pos_iono_north_t89  = np.empty((0,2))
    pos_iono_south_t89  = np.empty((0,2))
    pos_lmc_t89         = np.empty((0,9))
    pos_lstar_t89       = np.empty((0,9))
    pos_I_t89           = np.empty((0,9))
    pos_blocal_t89      = np.empty((0,))
    pos_beq_t89         = np.empty((0,))
    for i in date_finder(event_point)[0]:
        path=common_path+"ergsc/satellite/erg/orb/l3/t89/"+i.strftime('%Y')+"/"+i.strftime('%m')+"/"
        file=path+"erg_orb_l3_t89_"+i.strftime('%Y%m%d')+"_v02.cdf"
        if pathlib.Path(file).exists():
            data=cdflib.CDF(file)

            epoch               = np.concatenate((epoch,                data['epoch']))
            pos_eq_t89          = np.concatenate((pos_eq_t89,           data['pos_eq_t89']))
            pos_iono_north_t89  = np.concatenate((pos_iono_north_t89,   data['pos_iono_north_t89']))
            pos_iono_south_t89  = np.concatenate((pos_iono_south_t89,   data['pos_iono_south_t89']))
            pos_lmc_t89         = np.concatenate((pos_lmc_t89,          data['pos_lmc_t89']))
            pos_lstar_t89       = np.concatenate((pos_lstar_t89,        data['pos_lstar_t89']))
            pos_I_t89           = np.concatenate((pos_I_t89,            data['pos_I_t89']))
            pos_blocal_t89      = np.concatenate((pos_blocal_t89,       data['pos_blocal_t89']))
            pos_beq_t89         = np.concatenate((pos_beq_t89,          data['pos_beq_t89']))
        else: print("No file path:"+file)
    # pos_lstar_op[pos_b lstar_op<0]=0 #np.nan
    return epoch,pos_eq_t89,pos_iono_north_t89,pos_iono_south_t89,pos_lmc_t89,pos_lstar_t89,pos_I_t89,pos_blocal_t89,pos_beq_t89     

def erg_orb_def(event_point):
    epoch               = np.empty((0,)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pos_gse             = np.empty((0,3))
    pos_gsm             = np.empty((0,3))
    pos_sm              = np.empty((0,3))
    pos_rmlatmlt        = np.empty((0,3))
    pos_eq              = np.empty((0,2))
    pos_iono_north      = np.empty((0,2))
    pos_iono_south      = np.empty((0,2))
    pos_blocal          = np.empty((0,3))
    pos_blocal_mag      = np.empty((0,))
    pos_beq             = np.empty((0,3))
    pos_beq_mag         = np.empty((0,))
    pos_Lm              = np.empty((0,3))
    
    for i in date_finder(event_point)[0]:
        path=common_path+"ergsc/satellite/erg/orb/def/"+i.strftime('%Y')+"/"
        file=path+"erg_orb_l2_"+i.strftime('%Y%m%d')+"_v03.cdf"
        if pathlib.Path(file).exists():
            data=cdflib.CDF(file)

            epoch               = np.concatenate((epoch,            data['epoch'])) 
            pos_gse             = np.concatenate((pos_gse,          data['pos_gse'])) 
            pos_gsm             = np.concatenate((pos_gsm,          data['pos_gsm'])) 
            pos_sm              = np.concatenate((pos_sm,           data['pos_sm'])) 
            pos_rmlatmlt        = np.concatenate((pos_rmlatmlt,     data['pos_rmlatmlt'])) 
            pos_eq              = np.concatenate((pos_eq,           data['pos_eq'])) 
            pos_iono_north      = np.concatenate((pos_iono_north,   data['pos_iono_north'])) 
            pos_iono_south      = np.concatenate((pos_iono_south,   data['pos_iono_south'])) 
            pos_blocal          = np.concatenate((pos_blocal,       data['pos_blocal'])) 
            pos_blocal_mag      = np.concatenate((pos_blocal_mag,   data['pos_blocal_mag'])) 
            pos_beq             = np.concatenate((pos_beq,          data['pos_beq'])) 
            pos_beq_mag         = np.concatenate((pos_beq_mag,      data['pos_beq_mag'])) 
            pos_Lm              = np.concatenate((pos_Lm,           data['pos_Lm'])) 
            
        else: print("No file path:"+file)
    # pos_lstar_op[pos_b lstar_op<0]=0 #np.nan
    return epoch,pos_gse,pos_gsm,pos_sm,pos_rmlatmlt,pos_eq,pos_iono_north,pos_iono_south,pos_blocal,pos_blocal_mag,pos_beq,pos_beq_mag,pos_Lm     


def erg_hep_pa(event_point):
    hepl_epoch = np.empty((0,)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pa_l = np.empty((0,15))
    hepl_fedu = np.empty((0,16,15))
    heph_epoch = np.empty((0,)).astype(int) #astype(int)를 해야 시간계산이 제대로됨
    pa_h = np.empty((0,15))
    heph_fedu = np.empty((0,11,15))
    for i in date_finder(event_point)[0]:
        
        # path="/media/chondrite/1T-B/data/ergsc/satellite/erg/orb/l3/opq/"+i.strftime('%Y')+"/"+i.strftime('%m')+"/"
        # file=path+"erg_orb_l3_op_"+i.strftime('%Y%m%d')+"_v02.cdf"
        file=common_path+'inchun/20201025_HEP_daily_pa_01/HEP_pa_'+i.strftime('%Y%m%d')+'.npz'
        if pathlib.Path(file).exists():
            a=np.load(file, allow_pickle=True)
            a_hepl_epoch,a_pa_l,a_hepl_fedu,a_heph_epoch,a_pa_h,a_heph_fedu = a['arr_0']
            hepl_epoch  = np.concatenate((hepl_epoch,   a_hepl_epoch))
            pa_l        = np.concatenate((pa_l,         a_pa_l))
            hepl_fedu   = np.concatenate((hepl_fedu,    a_hepl_fedu))
            heph_epoch  = np.concatenate((heph_epoch,   a_heph_epoch))
            pa_h        = np.concatenate((pa_h,         a_pa_h))
            heph_fedu   = np.concatenate((heph_fedu,    a_heph_fedu))
            
        else: print("No file path:"+file)
    # pos_lstar_op[pos_lstar_op<0]=0 #np.nan
    return hepl_epoch, pa_l, hepl_fedu, heph_epoch, pa_h, heph_fedu



def read_CDF(CDF_variable,CDF_path):
    try: CDF_file = sorted(glob.glob(CDF_path+"*"+year+month+day+"*"),key=os.path.getctime)[0]
    except: CDF_file = "" #To avoid no cdf file
    if os.path.exists(CDF_file):
        print("/",CDF_variable,": OK ", end='')
        locals()[CDF_variable] = pycdf.CDF(CDF_file)  #read CDF
        keys = list(locals()[CDF_variable].keys())    #Extract CDF variables
        for i in range(len(keys)):
            try: globals()[CDF_variable+"_"+keys[i]]=locals()[CDF_variable][keys[i]][:]  #save CDF variables      
            except: globals()[CDF_variable+"_"+keys[i]]=locals()[CDF_variable][keys[i]]  #to avoid empty cdf variabls error    
            print(keys[i])     
    else: 
        print("/",CDF_variable,": NO ", end='')


storm=['2017-04-22/23:58:00']#['2017-03-27/14:45:00']#,'2017-04-04/07:06:00']#,'2017-04-22/23:58:00','2017-05-28/07:13:00','2017-07-16/15:51:00','2017-08-23/12:35:00','2017-08-31/11:48:00','2017-09-08/01:10:00','2017-09-13/00:12:00','2017-09-28/05:58:00','2017-10-14/05:36:00','2017-11-08/04:07:00','2017-11-21/06:55:00','2018-02-27/13:03:00','2018-03-18/21:45:00','2018-04-20/09:35:00','2018-05-06/02:32:00','2018-08-26/07:11:00','2018-09-11/10:12:00','2018-10-07/21:53:00','2018-11-05/06:03:00']#,'2019-05-11/03:03:00','2019-05-14/07:54:00','2019-08-05/09:12:00','2019-08-31/23:13:00','2019-09-02/00:45:00','2019-09-05/05:12:00','2019-09-27/23:08:00','2019-10-01/06:50:00']
before_event_day=1
lasting_event_day=5
common_path="/media/chondrite/PENCIL2/data/"
#common_path "/media/chondrite/1T-B/data/"

for event in storm:
    print(storm.index(event)+1,"/",len(storm),":",event)
    # moment=date_finder(event)[1]
    # moment_timelist=[moment.year,moment.month,moment.day-before_event_day,moment.hour,moment.minute,moment.second,0]
    # moment_timelist_end=[2000,1,lasting_event_day,0,0,0,0]
    # epoch_orb, pos_lstar_op, pos_blocal_op, pos_beq_op=append_orb(event)
    epoch_t89,pos_eq_t89,pos_iono_north_t89,pos_iono_south_t89,pos_lmc_t89,pos_lstar_t89,pos_I_t89,pos_blocal_t89,pos_beq_t89 = erg_orb_l3_t89(event)
    epoch_def,pos_gse,pos_gsm,pos_sm,pos_rmlatmlt,pos_eq,pos_iono_north,pos_iono_south,pos_blocal,pos_blocal_mag,pos_beq,pos_beq_mag,pos_Lm = erg_orb_def(event)
    hepl_epoch, pa_l, hepl_fedu, heph_epoch, pa_h, heph_fedu = erg_hep_pa(event)
    print("Read over event ", event)

#####

# # hepl_epoch, pa_l, hepl_fedu, heph_epoch, pa_h, heph_fedu = np.load('HEP_pa_20190701.npz')
# # a= np.load('HEP_pa_20190701.npz')
# a=np.load('20201025_HEP_daily_pa_01/HEP_pa_20170801.npz',allow_pickle=True)
# hepl_epoch,pa_l,hepl_fedu,heph_epoch,pa_h,heph_fedu = a['arr_0']

# # HEPL_epoch=a["HEPL_epoch"]
# # HEPL_PA=a["HEPL_PA"]
# # HEPL_FEDU=a["HEPL_FEDU"]
# # HEPH_epoch=a["HEPH_epoch"]
# # HEPH_PA=a["HEPH_PA"]
# # HEPH_FEDU =a["HEPH_FEDU"]
# plt.imshow(heph_fedu[:,:,7].T, origin="lower", interpolation="none",aspect='auto', cmap='jet',vmin=0,vmax=1e3)
# # plt.imshow(pa_h.T, origin="lower", interpolation="none",aspect='auto', cmap='jet')

# # plt.yticks(np.linspace(-0.5,len(y_boarder)-1.5,9+1),[0,1,2,3,4,5,6,7,8,9])
# # plt.xticks(np.linspace(-0.5,len(x_boarder)-1.5,lasting_event_day+1),range(-before_event_day,lasting_event_day-before_event_day+1))
# # plt.ylim( 2 *bin_size[1]/9-0.5, 8 *bin_size[1]/9-0.5 )
# plt.colorbar()
# # plt.xlabel('Day')  
# # plt.ylabel('PA: '+str(dir_ch)+' '+str(HEP_L_ene_LABL[ene_ch])+' keV\n L*')
# # plt.grid(True)
# plt.show()

# datetime.fromtimestamp(hepl_epoch[0])
# cdflib.cdfepoch.breakdown_tt2000(epoch_t89)[0]