import numpy as np
import sys,os
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import readPicks,extractFeatureData

# Script to scan through SEED files and extract feature values
featureDir='/home/andrewr/Desktop/Python/EventDetection_MM/FeatureArchive_hRes'
picksDir='/home/andrewr/Desktop/Python/Pickgot/WantedPicks/MM_ManualPicks'
outDir='/home/andrewr/Desktop/Python/EventDetection_MM/FeatureArchive_Manual_hRes_Try1_npy'
preArriv=2 # Number of samples before arrival to use for classification
postArriv=4 # Number of sampling after arrival to use for classification
L_I=0.12 # The sampling interval which was used during feature extraction
n_f=27 # Number of unique features used during feature extraction
archiveFileLen=3600 # How long a single archive file is in seconds

if not os.path.exists(outDir):
    os.makedirs(outDir)
  
def main():
    # For each pick file, extract the arrival-aligned features
    for aFile in sorted(os.listdir(picksDir)):
        print aFile
        aID=str(int(aFile.split('_')[0]))
        picks=readPicks(picksDir,aFile)
        stas,counts=np.unique([aPick.Sta for aPick in picks],return_counts=True)
        staTypes=[aPick.Sta+aPick.Type for aPick in picks]
        # Only read in the observation if both phases were picked
        for aSta in stas[np.where(counts==2)]:
            # If for some reason the same phase was picked twice, skip
            try:
                pAriv=picks[staTypes.index(aSta+'P')].Time
                sAriv=picks[staTypes.index(aSta+'S')].Time
            except:
                continue
            # If the S pick came before the P pick, skip
            if sAriv<pAriv:
                print aFile,aSta,'S-arrival was before P-arrival'
                continue
            # Take time slightly before and after the wanted times
            t1=pAriv-preArriv*(L_I+1)
            t2=sAriv+postArriv*(L_I+1)
            st=extractFeatureData(featureDir,t1,t2,[[aSta,'*']],archiveFileLen=archiveFileLen,fillVal=np.nan)
            if len(st)!=n_f:
                print aFile,aSta,'number of features does not match expected'
                continue
            elif st[0].stats.delta!=L_I:
                print aFile,aSta,'delta does not match expected'
                continue
            # Sort channels alphabetically
            st.sort(keys=['channel'])
            # Put the array together in the same way estimated arrivals are patched together
            pData,sData=[],[]
            for tr in st:
                pStart=int(round((pAriv-tr.stats.starttime)/L_I))
                sStart=int(round((sAriv-tr.stats.starttime)/L_I))
                pData.append(tr.data[pStart-preArriv:pStart+postArriv])
                sData.append(tr.data[sStart-preArriv:sStart+postArriv])
            pData=np.array(pData,dtype=float)
            sData=np.array(sData,dtype=float)
            staData=np.concatenate((pData.flatten(),sData.flatten()))
            np.save(outDir+'/'+aID.zfill(6)+'_'+aSta+'_FeatureData',staData)
#            print staData.shape

if __name__ == "__main__":
    main()