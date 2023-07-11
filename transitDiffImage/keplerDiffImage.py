import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
import pprint
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
import copy
import requests
import os
import subprocess
from astropy.io import fits

def get_koi_data(koi):
    selectStr = "kepid,koi_tce_plnt_num,kepoi_name,koi_kepmag,koi_period,koi_time0bk,koi_duration,koi_depth,ra,dec,koi_pdisposition,"
    whereStr = "kepoi_name like " + "'K{0:05.0f}%25'".format(koi)
#    print(whereStr)
    urlDr25Koi = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr25_koi&select=" + selectStr + "&where=" + whereStr
#    print(urlDr25Koi)
    r = requests.get(urlDr25Koi)
    tt = r.content.decode("utf-8").split('\n')
#    print(tt)

    otherPlanetList = []
    for i in range(len(tt)-2): # -2 because a) skip first line and b) split('\n') leaves a trailing empty line
        archiveData = np.array(tt[i+1].split(","))

        if archiveData[2] == "K{0:08.2f}".format(koi):
            koiData = {}
            koiData["kepid"] = int(archiveData[0])
            koiData["tceNum"] = int(archiveData[1])
            koiData["koiNum"] = archiveData[2]
            if archiveData[3]!='':
                koiData["kepmag"] = float(archiveData[3])
            else:
                koiData["kepmag"] = -1
            koiData["period"] = float(archiveData[4])
            koiData["epoch"] = float(archiveData[5])
            koiData["durationHours"] = float(archiveData[6])
            if archiveData[7]!='':
                koiData["observedDepth"] = float(archiveData[7])/1e6
            else:
                koiData["observedDepth"] = -1
            koiData["ra"] = float(archiveData[8])
            koiData["dec"] = float(archiveData[9])
            koiData["disposition"] = archiveData[10]
        else:
            planetData = {}
            planetData["koiNum"] = archiveData[2]
            planetData["period"] = float(archiveData[4])
            planetData["epoch"] = float(archiveData[5])
            planetData["durationHours"] = float(archiveData[6])
            otherPlanetList.append(planetData)
    koiData["otherPlanetList"] = otherPlanetList
    c = SkyCoord(ra=koiData["ra"]*u.degree, dec=koiData["dec"]*u.degree, frame='icrs')
    koiData["galacticLatitude"] = c.galactic.b.degree

    pprint.pprint(koiData)
    return koiData


def get_dv_model(koiData, wgetFileName):
    # grep the file for this kepid
    # a reader is returned
    print(wgetFileName)
    a =  subprocess.Popen("grep "+str(koiData["kepid"])+" "+wgetFileName+" | grep _dvt.fits",
                          shell=True, stdout=subprocess.PIPE).stdout
    print(a)
    # decode the output, and split with newline because the same kepid can appear multiple times
    gs = a.read().decode().split("\n")[0]
    print(gs)
    # get the fits file URL, which is the fourth term in the output line
    urlStr = gs.split(" ")[3]
    # get the file
    getStr = "wget -q -O tempFits.fits " + urlStr
#    print(getStr)
    os.system(getStr)
    
    fitsData = fits.open("tempFits.fits")
    # there are several elements:
    # element 0 is overall information
    # followed by one element per TCE
    # so the TCE number correctly indexes into the fits file
    dvModel = {}
    dvModel["model"] = np.zeros(len(fitsData[koiData["tceNum"]].data))
    dvModel["time"] = np.zeros(len(fitsData[koiData["tceNum"]].data))
    for i in range(len(fitsData[koiData["tceNum"]].data)):
        # BKJD time is the first field
        dvModel["time"][i] = fitsData[koiData["tceNum"]].data[i][0]
        # the DV lightcurve model is the 9th field
        dvModel["model"][i] = fitsData[koiData["tceNum"]].data[i][8]

    return dvModel

def mag2b(mag):
    return (100**(1/5))**(-mag)

def mag2flux(mag):
    flux12 = 3.6e8/(30*60)
    return flux12*mag2b(mag)/mag2b(12)

def get_gaia_catalog(tpf, mjd, rdp, supplementalCatalog = None):
    # compute mjd for the Gaia epoch J2016 = 2016-01-01T00:00:00
    t = Time("2016-01-01T00:00:00", format='isot', scale='utc')
    mjdJ2016 = t.mjd

    # Get the Gaia catalog entries for stars near the KOI.
    # Use half the diagonal of the pixel image for the search radius, assuming 4 arcsec per pixel.
    Gaia.ROW_LIMIT = -1
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    searchCenter = SkyCoord(ra=tpf.ra, dec=tpf.dec, unit=(u.degree, u.degree), frame='icrs')
    searchRadius = u.Quantity((np.linalg.norm([tpf.shape[1], tpf.shape[2]]))*4/3600/2 , u.deg) # radius in degrees
    j = Gaia.cone_search_async(searchCenter, searchRadius)
    gaiaCatalog = j.get_results()
    

    # correct star position for proper motion
    dRa = np.zeros(len(gaiaCatalog))
    dDec = np.zeros(len(gaiaCatalog))
    correctedRa = np.zeros(len(gaiaCatalog))
    correctedDec = np.zeros(len(gaiaCatalog))
    mas2deg = 1/(3600*1000)
    dt = (mjd - mjdJ2016)/365
    dRa = mas2deg*dt*gaiaCatalog["pmra"]/np.cos(gaiaCatalog["dec"]*np.pi/180)
    dDec = mas2deg*dt*gaiaCatalog["pmdec"]
    pmraSigma = np.array(gaiaCatalog["pmra_error"])
    pmraSigma[np.isnan(pmraSigma)] = 0
    pmdecSigma = np.array(gaiaCatalog["pmdec_error"])
    pmdecSigma[np.isnan(pmdecSigma)] = 0
    gaiaCatalog["correctedGaiaRa"] = gaiaCatalog["ra"] + dRa
    gaiaCatalog["correctedGaiaRaSigma"] = np.sqrt((mas2deg*gaiaCatalog["ra_error"])**2
                                   + (mas2deg*dt*pmraSigma/np.cos(gaiaCatalog["dec"]*np.pi/180))**2)
    gaiaCatalog["correctedGaiaDec"] =  gaiaCatalog["dec"] + dDec
    gaiaCatalog["correctedGaiaDecSigma"] = np.sqrt((mas2deg*np.array(gaiaCatalog["dec_error"]))**2
                                    + (mas2deg*dt*pmdecSigma)**2)
    if supplementalCatalog is not None:
        for s in range(len(supplementalCatalog)):
            refStar = supplementalCatalog[s]["referenceStarIndex"]
            newRow = copy.deepcopy(gaiaCatalog[0])
            for i in range(len(newRow)):
                newRow[i] = 0

            newRow["ra"] = gaiaCatalog[refStar]["ra"] + supplementalCatalog[s]["deltaRaArcsec"]/3600/np.sin(gaiaCatalog[refStar]["dec"]*np.pi/180)
            newRow["dec"] = gaiaCatalog[refStar]["dec"] + supplementalCatalog[s]["deltaDecArcsec"]/3600
            newRow["correctedGaiaRa"] = gaiaCatalog[refStar]["correctedGaiaRa"] + supplementalCatalog[s]["deltaRaArcsec"]/3600/np.sin(gaiaCatalog[refStar]["dec"]*np.pi/180)
            newRow["correctedGaiaDec"] = gaiaCatalog[refStar]["correctedGaiaDec"] + supplementalCatalog[s]["deltaDecArcsec"]/3600
            newRow["source_id"] = supplementalCatalog[s]["source_id"]
            newRow["phot_g_mean_mag"] = supplementalCatalog[s]["phot_g_mean_mag"]
            newRow["phot_g_mean_flux"] = mag2flux(newRow["phot_g_mean_mag"])
            gaiaCatalog.add_row(newRow)

    # compute the pixel locations of the stars
#    rdp = raDec2Pix.raDec2PixClass("../raDec2Pix/Kepler-RaDex2Pix/raDec2PixDir")
    [gaiaMod, gaiaOut, gaiaRow, gaiaCol] = rdp.ra_dec_2_pix(gaiaCatalog["correctedGaiaRa"],
                                                            gaiaCatalog["correctedGaiaDec"], mjd)
    gaiaCatalog["gaiaSigma"] = np.array(np.sqrt(gaiaCatalog["correctedGaiaRaSigma"]**2
                                                + gaiaCatalog["correctedGaiaDecSigma"]**2)*3600/3.98) # degrees to pixels
#    print("gaiaSigma = " + str(gaiaCatalog["gaiaSigma"]))
    [gaiaCatalog["targetMod"], gaiaCatalog["targetOut"], \
         gaiaCatalog["targetRow"], gaiaCatalog["targetCol"]] \
        = rdp.ra_dec_2_pix(tpf.ra, tpf.dec, mjd)
    
    # remove the stars that are not on collected pixels
    # compute the rows and colums of the pixels in tpf
    pixRow = np.zeros((tpf.shape[1], tpf.shape[2]))
    pixCol = np.zeros((tpf.shape[1], tpf.shape[2]))
    for r in range(tpf.shape[1]):
        for c in range(tpf.shape[2]):
            pixRow[r,c] = tpf.row + r
            pixCol[r,c] = tpf.column + c
    goodStarIndex = []
    for s in range(len(gaiaCatalog)):
        goodStarIndex.append(s)

#     goodStarIndex = []
#     for s in range(len(gaiaCatalog)):
#         if (np.in1d(np.round(gaiaRow[s]), pixRow.flatten()) & np.in1d(np.round(gaiaCol[s]), pixCol.flatten())) \
#             & (gaiaCatalog["phot_g_mean_mag"][s] < 18):
#             goodStarIndex.append(s)
#         else:
#             print("rejecting star " + str(s))
    gaiaCatalog = gaiaCatalog[goodStarIndex]
    gaiaCatalog["flux"] = mag2flux(gaiaCatalog["phot_g_mean_mag"])
    gaiaCatalog["mod"] = gaiaMod[goodStarIndex]
    gaiaCatalog["out"] = gaiaOut[goodStarIndex]
    gaiaCatalog["row"] = gaiaRow[goodStarIndex]
    gaiaCatalog["col"] = gaiaCol[goodStarIndex]

    return gaiaCatalog

class keplerDiffImage:

    def __init__(self, tpf, koiData, allowedBadCadences = 3, transitThreshold = 0.75, qMask = lk.KeplerQualityFlags.DEFAULT_BITMASK):
        self.tpf = tpf
        self.koiData = koiData
        self.allowedBadCadences = allowedBadCadences
        self.transitThreshold = transitThreshold
        
        self.dt = np.min(np.diff(self.tpf.time.bkjd))
        self.qMask = qMask


    def find_transit_times(self, koiData):
        nTransit = np.round((self.tpf.time.bkjd - koiData["epoch"])/koiData["period"]).astype(int)
        transitTimes = np.unique(koiData["epoch"] + koiData["period"] * nTransit)
        transitIndex = np.array([np.abs(self.tpf.time.bkjd - t).argmin() for t in transitTimes])
        bufferRatio = 0.5
        flagGaps = np.abs(self.tpf.time[transitIndex].bkjd - transitTimes) > bufferRatio*self.dt
    #    for i in np.nonzero(flagGaps)[0]:
    #        print("large cadence difference: " + str(self.tpf.time[transitIndex][i] - transitTimes[i]))
        transitTimes = transitTimes[~flagGaps]
        transitIndex = transitIndex[~flagGaps]
        return transitTimes, transitIndex

    def find_transits(self, transitModel = None):

        self.transitTimes, self.transitIndex = self.find_transit_times(self.koiData)
        self.inTransitIndices = []
        self.outTransitIndices = []

        self.durationDays = self.koiData["durationHours"]/24;
        transitAverageDurationDays = 0.8*self.durationDays/2;

        outTransitBuffer = self.dt + self.durationDays
        expectedInTransitLength = np.floor(2*transitAverageDurationDays/self.dt)

        # avoid transit indices in transits from other KOIs in the same system
        self.inOtherTransitIndices = np.array([])
        for otherPlanet in self.koiData["otherPlanetList"]:
#            print("other planet " + otherPlanet["koiNum"])
            otherTransitTimes, otherTransitIndex = self.find_transit_times(otherPlanet)
            
            durationDays = otherPlanet["durationHours"]/24;
            transitAverageDurationDays = 1.2*durationDays/2;
            for i in otherTransitIndex:
                oti = np.nonzero((np.abs(self.tpf.time[i].bkjd - self.tpf.time.bkjd) < transitAverageDurationDays))[0]
    #            print(self.tpf.time[oti])
                self.inOtherTransitIndices = np.append(self.inOtherTransitIndices,oti)

        self.inOtherTransitIndices = np.array(self.inOtherTransitIndices).astype(int)
    #    print("all other transit times: " + str(self.tpf.time[inOtherTransitIndices]))

        self.inOtherTransitFlag = np.zeros(self.tpf.quality.shape)
        self.inOtherTransitFlag[self.inOtherTransitIndices] = 1
        self.inOtherTransitFlag = np.array(self.inOtherTransitFlag).astype(int)

    #    print("sum of quality: " + str(np.sum(self.tpf.quality&qMask)))
        self.cadenceQuality = self.tpf.quality&self.qMask
        self.cadenceQuality[self.inOtherTransitIndices] = 1 # force in other transit cadences to 1
    #    print("sum of quality or in other transit: " + str(np.sum(self.cadenceQuality)))

        if len(self.transitTimes) == 0:
            return []
        nBadCadences = []
        DiffImageDataList = []
        goodTransitList = []
#        print(self.transitIndex)
        
        for i in self.transitIndex:
#            print("transit " + str(i))
            if transitModel is None:
                thisTransitInIndices = np.nonzero(
                  (np.abs(self.tpf.time[i].bkjd - self.tpf.time.bkjd) < transitAverageDurationDays))[0]
            else:
                transitTreshold = self.transitThreshold*np.min(transitModel["model"])
                # extract the transit model around this transit time
#                print(transitModel["time"])
#                print(self.tpf.time[i].bkjd)
#                print(transitTreshold)
#                print(self.durationDays)
                mt = transitModel["time"]
                modelIndex = (mt > self.tpf.time[i].bkjd - 2*self.durationDays) & (mt < self.tpf.time[i].bkjd + 2*self.durationDays) & (transitModel["model"] < transitTreshold)
#                print(sum(modelIndex))
                ThisTransitInTimes = mt[modelIndex]
#                print(len(ThisTransitInTimes))
#                print(ThisTransitInTimes)
#                print([np.round(self.tpf.time.bkjd, 6), np.round(ThisTransitInTimes, 6)])
#                print(np.intersect1d(np.round(self.tpf.time.bkjd, 6), np.round(ThisTransitInTimes, 6)))
                _, thisTransitInIndices, _ = np.intersect1d(np.round(self.tpf.time.bkjd, 6), np.round(ThisTransitInTimes, 6), return_indices=True)
#                print(thisTransitInIndices)

    #        print([self.tpf.time[i] - outTransitBuffer, self.tpf.time[i], self.tpf.time[i] + outTransitBuffer])
    #        print([self.dt*len(thisTransitInIndices), self.dt, len(thisTransitInIndices)])
            thisTransitOutIndices = np.nonzero(
              (np.abs((self.tpf.time[i].bkjd - outTransitBuffer) - self.tpf.time.bkjd) < self.dt*len(thisTransitInIndices)/2)
              | (np.abs((self.tpf.time[i].bkjd + outTransitBuffer) - self.tpf.time.bkjd) < self.dt*len(thisTransitInIndices)/2))[0]
    #        print(thisTransitOutIndices)
             # this is a mirror version of the above lines
    #        thisTransitOutIndices = np.nonzero(
    #          (np.abs(self.tpf.time[i] - self.tpf.time) > (outTransitBuffer - transitAverageDurationDays))
    #          & (np.abs(self.tpf.time[i] - self.tpf.time) < (outTransitBuffer + transitAverageDurationDays)))[0]
            
            # filter out bad cadences
            thisTransitNumBadCadences = np.sum(self.cadenceQuality[thisTransitInIndices] > 0) + np.sum(self.cadenceQuality[thisTransitOutIndices] > 0)
    #        print("this in transit cadence quality = " + str(self.cadenceQuality[thisTransitInIndices]))
    #        print("this in transit times = " + str(self.tpf.time[thisTransitInIndices]))
    #        print("this out transit cadence quality = " + str(self.cadenceQuality[thisTransitOutIndices]))
    #        print("this out transit times = " + str(self.tpf.time[thisTransitOutIndices]))
    #        print("thisTransitNumBadCadences = " + str(thisTransitNumBadCadences))

    #        if (len(thisTransitInIndices) < expectedInTransitLength) | (len(thisTransitOutIndices) < 2*expectedInTransitLength):
    #          continue
            nBadCadences.append(thisTransitNumBadCadences)
#            print([thisTransitNumBadCadences,self.allowedBadCadences])
            if thisTransitNumBadCadences < self.allowedBadCadences:
                thisTransitInIndices = thisTransitInIndices[self.cadenceQuality[thisTransitInIndices] == 0].tolist()
                thisTransitOutIndices = thisTransitOutIndices[self.cadenceQuality[thisTransitOutIndices] == 0].tolist()
#                print("making difference image")
#                print("thisTransitInIndices = " + str(thisTransitInIndices))
#                print("thisTransitOutIndices = " + str(thisTransitOutIndices))
                thisDiffImage = self.make_difference_image(thisTransitInIndices, thisTransitOutIndices)
                if thisDiffImage is not None:
                    DiffImageDataList.append(thisDiffImage)
            
                self.inTransitIndices.append(thisTransitInIndices)
                self.outTransitIndices.append(thisTransitOutIndices)
                
                goodTransitList.append(i)
        if len(DiffImageDataList) == 0:
            # we have no cadences that meet the allowedBadCadences, so pick the least worse
            leastBadTransitIndex = np.argmin(nBadCadences)
#            print("nBadCadences = " + str(nBadCadences))

            thisTransitInIndices = thisTransitInIndices[self.cadenceQuality[leastBadTransitIndex] == 0].tolist()
#            print("thisTransitInIndices = " + str(thisTransitInIndices))
            thisTransitOutIndices = thisTransitOutIndices[self.cadenceQuality[leastBadTransitIndex] == 0].tolist()
#            print("thisTransitOutIndices = " + str(thisTransitOutIndices))
            if len(thisTransitInIndices) > 0 & (len(thisTransitOutIndices) > 0):
                thisDiffImage = self.make_difference_image(thisTransitInIndices[0], thisTransitOutIndices[0])
                if thisDiffImage is not None:
                    DiffImageDataList.append(thisDiffImage)
            
                self.inTransitIndices.append(thisTransitInIndices)
                self.outTransitIndices.append(thisTransitOutIndices)
            
            goodTransitList = DiffImageDataList
            
            print("Warning: all transits have too many bad cadences, picking transit " + str(leastBadTransitIndex) + " with " + str(nBadCadences[leastBadTransitIndex]) + " bad cadences.")

        if len(DiffImageDataList) == 0:
            return []
            
#        print("length of diffImageDataList = " + str(len(DiffImageDataList)))
        self.diffImageData = {}
        self.diffImageData["diffImage"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["diffImageSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["diffSNRImage"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["meanInTransit"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["meanInTransitSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["meanOutTransit"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        self.diffImageData["meanOutTransitSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        for i in range(len(DiffImageDataList)):
            self.diffImageData["diffImage"] += DiffImageDataList[i]["diffImage"]
            self.diffImageData["diffImageSigma"] += DiffImageDataList[i]["diffImageSigma"]**2
            self.diffImageData["meanInTransit"] += DiffImageDataList[i]["meanInTransit"]
            self.diffImageData["meanInTransitSigma"] += DiffImageDataList[i]["meanInTransitSigma"]**2
            self.diffImageData["meanOutTransit"] += DiffImageDataList[i]["meanOutTransit"]
            self.diffImageData["meanOutTransitSigma"] += DiffImageDataList[i]["meanOutTransitSigma"]**2
        self.diffImageData["diffImage"] /= len(DiffImageDataList)
        self.diffImageData["diffImageSigma"] = np.sqrt(self.diffImageData["diffImageSigma"])/len(DiffImageDataList)
        self.diffImageData["meanInTransit"] /= len(DiffImageDataList)
        self.diffImageData["meanInTransitSigma"] = np.sqrt(self.diffImageData["meanInTransitSigma"])/len(DiffImageDataList)
        self.diffImageData["meanOutTransit"] /= len(DiffImageDataList)
        self.diffImageData["meanOutTransitSigma"] = np.sqrt(self.diffImageData["meanOutTransitSigma"])/len(DiffImageDataList)
        self.diffImageData["diffSNRImage"] = self.diffImageData["diffImage"]/self.diffImageData["diffImageSigma"]

        self.inTransitIndices = np.unique(sum(np.array(self.inTransitIndices).tolist(), [])).astype(int)
        self.outTransitIndices = np.unique(sum(np.array(self.outTransitIndices).tolist(), [])).astype(int)

        return self.diffImageData

    def make_difference_image(self, inTransitIndices, outTransitIndices):
        if np.all(np.isnan(self.tpf.flux[inTransitIndices,:,:])) | np.all(np.isnan(self.tpf.flux[outTransitIndices,:,:])):
            print("make_difference_image: all flux is Nan, returning None")
            return None
        
        meanInTransit = np.nanmean(self.tpf.flux[inTransitIndices,::-1,:].value, axis=0)
        meanInTransitSigma = np.sqrt(np.nanmean(self.tpf.flux_err[inTransitIndices,::-1,:].value**2, axis=0)/len(inTransitIndices))
        meanOutTransit = np.nanmean(self.tpf.flux[outTransitIndices,::-1,:].value, axis=0)
        meanOutTransitSigma = np.sqrt(np.nanmean(self.tpf.flux_err[outTransitIndices,::-1,:].value**2, axis=0)/len(outTransitIndices))
        diffImage = meanOutTransit-meanInTransit
        diffImageSigma = np.sqrt((meanInTransitSigma**2)+(meanOutTransitSigma**2))
        diffSNRImage = diffImage/diffImageSigma

        diffImageData = {}
        diffImageData["diffImage"] = diffImage
        diffImageData["diffImageSigma"] = diffImageSigma
        diffImageData["diffSNRImage"] = diffSNRImage
        diffImageData["meanInTransit"] = meanInTransit
        diffImageData["meanInTransitSigma"] = meanInTransitSigma
        diffImageData["meanOutTransit"] = meanOutTransit
        diffImageData["meanOutTransitSigma"] = meanOutTransitSigma
        
        return diffImageData

    def draw_transits_lc(self, transitIndex = None, windowSize = None, aperture = None):
        if aperture is None:
            lc = self.tpf.to_lightcurve()
        else:
            lc = self.tpf.to_lightcurve(aperture_mask=aperture.astype(bool))

        windowBuff = 40;
        plt.figure(figsize=(15, 5));
        plt.plot(lc.time.bkjd, lc.flux.value, label="flux")
        plt.plot(lc.time[self.inTransitIndices].bkjd, lc.flux[self.inTransitIndices].value, 'd', ms=10, alpha = 0.6, label="in transit")
        plt.plot(lc.time[self.outTransitIndices].bkjd, lc.flux[self.outTransitIndices].value, 'o', ms=10, alpha = 0.5, label="out of transit")
        plt.plot(lc.time[(lc.quality&self.qMask)>0].bkjd, lc.flux[(lc.quality&self.qMask)>0].value, 'rx', ms=10, label="quality problems")
        plt.plot(lc.time[self.inOtherTransitIndices].bkjd, lc.flux[self.inOtherTransitIndices].value, 'y+', ms=20, label="other transit")
            
        if transitIndex is not None:
            timeStart = lc.time[transitIndex].bkjd - windowSize
            timeEnd = lc.time[transitIndex].bkjd + windowSize
            plt.xlim(timeStart, timeEnd)
            indexInWindow = (lc.time.bkjd > timeStart) & (lc.time.bkjd < timeEnd)
            meanDataInWindow = np.mean(lc.flux[indexInWindow].value)
            yMin = meanDataInWindow - 1.5*(meanDataInWindow - np.min(lc.flux[indexInWindow].value))
            yMax = meanDataInWindow - 1.5*(meanDataInWindow - np.max(lc.flux[indexInWindow].value))
            plt.ylim(yMin, yMax)
        plt.legend(fontsize=18)
        plt.tight_layout();
        plt.ylabel("flux");
        plt.xlabel("time (BKJD)");
        plt.title("flux light curve and in/out transit cadences", fontsize=20);
        
    def draw_each_transit(self, windowSize = None, aperture = None):
        if windowSize is None:
            windowSize = 3*self.koiData["durationHours"]/24
        for ti in self.transitIndex:
            self.draw_transits_lc(transitIndex=ti, windowSize=windowSize, aperture = aperture)

        
        
    def draw_difference_image(self, diffImageData, quarter, extent):
        fig = plt.figure(figsize=plt.figaspect(0.3));
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(diffImageData["diffImage"], extent=extent, cmap='jet')
        plt.colorbar()
        plt.title("diff image in quarter " + str(quarter));

        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(diffImageData["diffSNRImage"], extent=extent, cmap='jet')
        plt.colorbar()
        plt.title("SNR diff image in quarter " + str(quarter));

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(diffImageData["meanOutTransit"], extent=extent, cmap='jet')
        plt.colorbar()
        plt.title("Direct image in quarter " + str(quarter));
        
