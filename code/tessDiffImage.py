import os
import glob
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import astropy.table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from scipy.optimize import minimize
import tess_stars2px
import pickle
from scanf import scanf

import tess_stars2px as trdp
import barycentricCorrection as bc

flux12 = 3.6e8/(30*60)

def mag2b(mag):
    return (100**(1/5))**(-mag)

def mag2flux(mag):
    return flux12*mag2b(mag)/mag2b(12)

def get_tic(ra, dec, radiusDegrees):
    c = SkyCoord(ra, dec, unit=('deg', 'deg'), frame='icrs')
    return Catalogs.query_region(c, radius=radiusDegrees, catalog="TIC")

def add_quality_flags(pixelData, cadence, quality):
    qlpInCadences = np.isin(cadence, pixelData["cadenceNumber"])
    gotQlpFlag = np.isin(pixelData["cadenceNumber"], cadence[qlpInCadences])
    pixelData["quality"][gotQlpFlag] = pixelData["quality"][gotQlpFlag] + quality[qlpInCadences]
    return pixelData
    
def pix_distance(raDec, sector, camera, ccd, desiredCol, desiredRow):
    
    _, _, _, _, _, _, col, row, scinfo = tess_stars2px.tess_stars2px_function_entry(0,
                                                                              raDec[0],
                                                                              raDec[1],
                                                                              aberrate=True,
                                                                              trySector=sector)
    dCol = col[0] - desiredCol
    dRow = row[0] - desiredRow
    return dCol*dCol + dRow*dRow

def pix_to_ra_dec(sector, cam, ccd, col, row):
    seedRa, seedDec, scinfo = tess_stars2px.tess_stars2px_reverse_function_entry(sector, cam, ccd, col, row)

    seed = np.array([seedRa, seedDec])
    r = minimize(pix_distance, seed, method="L-BFGS-B",
                     args = (sector, cam, ccd, col, row))
    return r.x

class tessDiffImage:
    def __init__(self,
            ticData,
            spiceFileLocation = ".",
            nPixOnSide = 21,
            dMagThreshold = 4,
            allowedBadCadences = 0,
            allowedInTransitLossFraction = 0.8,
            maxOrbits = 116,
            cleanFiles = True,
            outputDir = "./",
            qlpFlagsLocation = None):
            
        self.ticData = ticData
        self.spiceFileLocation = spiceFileLocation
        self.nPixOnSide = nPixOnSide
        self.dMagThreshold = dMagThreshold
        self.allowedBadCadences = allowedBadCadences
        self.allowedInTransitLossFraction = allowedInTransitLossFraction
        self.maxOrbits = maxOrbits
        self.ticName = "tic" + str(self.ticData["id"])
        self.sectorList = []
        self.outputDir = outputDir
        self.cleanFiles = cleanFiles
        self.qlpFlagsLocation = qlpFlagsLocation


        self.baryCorrector = bc.barycentricCorrection(self.spiceFileLocation)


    def make_ffi_difference_image(self, thisPlanet=None, allowedBadCadences = None, drawImages = False):
    
        # check to see if the sector pixel files already exist
        sectorList = []
        if thisPlanet is not None:
            planetStr = self.ticData["planetData"][thisPlanet]["planetID"]
        else:
            planetStr = "*"
        if self.ticData["sector"] is not None:
            sectorStr = str(self.ticData["sector"])
        else:
            sectorStr = "*"
        ff = glob.glob(os.path.join(self.outputDir, self.ticName, "imageData_" + planetStr + "_sector" + sectorStr + ".pickle"))
        if len(ff) > 0:
            for f in ff:
                sector = scanf("sector%d.pickle", os.path.basename(f).split("_")[-1])[0]
                print("Found existing sector " + str(sector) + " file at " + f)
                sectorList.append(sector)
        if len(sectorList) > 0:
            self.sectorList = sectorList
            return
        
        # we need to make the difference images
        if allowedBadCadences is None:
            allowedBadCadences = self.allowedBadCadences
            
        fitsList = self.get_tess_cut()
        for fitsFile in fitsList:
            print(fitsFile)
            pixelData = self.get_cadence_data(fitsFile)
            # check that this is the camera and sector that we want here
            if (self.ticData["cam"] is not None) & (self.ticData["sector"] is not None) & ((self.ticData["sector"] != pixelData["sector"]) | (self.ticData["cam"] != pixelData["camera"])):
                print("not on specified camera or sector")
                continue

            if self.qlpFlagsLocation is not None:
                sectorQflags = np.array([])
                # Get QLP quality flags
                orbit1, orbit2 = pixelData["sector"]*2+7, pixelData["sector"]*2+8
                if orbit1 <= self.maxOrbits:
                    cam = pixelData["camera"]
                    ccd = pixelData["ccd"]
                    orb1File = f"orbit{orbit1}/orbit{orbit1}cam{cam}ccd{ccd}_qflag.txt"
                    orb2File = f"orbit{orbit2}/orbit{orbit2}cam{cam}ccd{ccd}_qflag.txt"
    #                print(orb1File)
    #                print(orb2File)
                    sectorQflags = np.loadtxt(self.qlpFlagsLocation + orb1File)
    #                print(sectorQflags.shape)
                    sectorQflags = np.concatenate((sectorQflags, np.loadtxt(self.qlpFlagsLocation + orb2File))).astype(int)
    #                print(sectorQflags.shape)

                    qlp_cad = sectorQflags[:,0]
                    qlp_flag = sectorQflags[:,1]
    #                print("len(qlp_flag) = " + str(len(qlp_flag)))
    #                qlpInCadences = np.isin(qlp_cad, tsc_cad)
    #                gotQlpFlag = np.isin(tsc_cad, qlp_cad[qlpInCadences])
    #                print("sum(gotQlpFlag) = " + str(sum(gotQlpFlag)) + " of " + str(len(gotQlpFlag)))
    #                pixelData["quality"][gotQlpFlag] = pixelData["quality"][gotQlpFlag] + qlp_flag[qlpInCadences]
                    pixelData = add_quality_flags(pixelData, qlp_cad, qlp_flag)
                else:
                    print("No QLP quality flags for orbit " + str(orbit1))


            if self.ticData["qualityFlags"] is not None:
                # self.ticData["qualityFlags"] has form [cadence list, quality list]
                pixelData = add_quality_flags(pixelData, self.ticData["qualityFlags"][0], self.ticData["qualityFlags"][1])

            catalogData = self.make_stellar_scene(pixelData)
            if len(catalogData) == 0:
                continue
            
            if thisPlanet is None:
                for p, planetData in enumerate(self.ticData["planetData"]):
                    planetData["planetIndex"] = p
                    planetData["planetIndex"] = p
                    self.make_planet_difference_image(self.ticData, planetData, pixelData, catalogData,
                            allowedBadCadences = allowedBadCadences, drawImages = drawImages)
            else:
                planetData = self.ticData["planetData"][thisPlanet]
                planetData["planetIndex"] = thisPlanet
                planetData["planetIndex"] = thisPlanet
                print("making difference image for sector " + str(pixelData["sector"]))
                self.make_planet_difference_image(planetData, pixelData, catalogData, allowedBadCadences = allowedBadCadences, drawImages = drawImages)
            self.sectorList.append(pixelData["sector"])

        #       print('rm ' + self.ticName + '/*.fits')
        if self.cleanFiles:
            os.system('rm ' + os.path.join(self.outputDir, self.ticName, '*.fits'))

    def make_planet_difference_image(self, planetData, pixelData, catalogData, allowedBadCadences = None, drawImages = False):
        if allowedBadCadences is None:
            allowedBadCadences = self.allowedBadCadences
            
        # for each planet, make an array of cadences that are in the transits of other planets
        inOtherTransitIndices = [];
        for pi, otherPlanet in enumerate(self.ticData["planetData"]):
            if pi == planetData["planetIndex"]:
                continue
            else:
                transitTimes, transitIndex = self.find_transit_times(pixelData, otherPlanet)
                
                durationDays = otherPlanet["durationHours"]/24;
                transitAverageDurationDays = durationDays/2;
                for i in transitIndex:
                    thisTransitInIndices = np.argwhere(
                        np.abs(pixelData["time"][i] - pixelData["time"]) < transitAverageDurationDays)
                    inOtherTransitIndices = np.append(inOtherTransitIndices, thisTransitInIndices)
                    
        inOtherTransitIndices = np.array(inOtherTransitIndices).astype(int)
        pixelData["inOtherTransit"] = np.zeros(pixelData["quality"].shape, dtype=int)
        pixelData["inOtherTransit"][inOtherTransitIndices] = 1
        
        pixelData["quality"] += pixelData["inOtherTransit"]

        inTransitIndices, outTransitIndices, transitIndex, diffImageData = self.find_transits(pixelData, planetData, allowedBadCadences = allowedBadCadences)
#        print(inTransitIndices)
#        print(outTransitIndices)
#        print(transitIndex)
#        diffImageData = self.make_difference_image(pixelData, inTransitIndices, outTransitIndices)
        if len(diffImageData) > 0:
            if drawImages:
                self.draw_difference_image(diffImageData, pixelData, planetData, catalogData)
                self.draw_lc_transits(pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex)

        f = open(os.path.join(self.outputDir, self.ticName, "imageData_" + planetData["planetID"] + "_sector" + str(pixelData["sector"]) + ".pickle"), 'wb')
        pickle.dump([diffImageData, catalogData, pixelData, inTransitIndices, outTransitIndices, transitIndex, planetData], f, pickle.HIGHEST_PROTOCOL)
        f.close()


    def get_tess_cut(self, fitsNum = 0):
        ticNumber = self.ticData["id"]
        ra = self.ticData["raDegrees"]
        dec = self.ticData["decDegrees"]
        sector = self.ticData["sector"]
        ticName = self.ticName
        dirStr = os.path.join(self.outputDir, ticName)
        if not os.path.exists(dirStr):
            os.system('mkdir ' + dirStr)
        if sector is None:
            zipStr = os.path.join(self.outputDir, ticName + '.zip')
        else:
            zipStr = os.path.join(self.outputDir, ticName + "_s" + str(sector) + '.zip')
        if not os.path.exists(zipStr):
            if sector is None:
                curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                            + str(ra) + '&dec=' + str(dec) + '&y=' + str(self.nPixOnSide) + '&x=' + str(self.nPixOnSide) \
                            + '" --output ' + zipStr
            else:
                curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                            + str(ra) + '&dec=' + str(dec) + '&y=' + str(self.nPixOnSide) + '&x=' + str(self.nPixOnSide) + '&sector=' + str(sector) \
                            + '" --output ' + zipStr
            print(curlStr)
            os.system(curlStr)
            os.system('unzip ' + zipStr + ' -d ' + dirStr)
        elif not os.path.exists(dirStr): # check for unexpanded zip file without a directory
            os.system('unzip ' + zipStr + ' -d ' + dirStr)

        if self.cleanFiles:
            os.system('rm ' + zipStr)
        fitsList = glob.glob(os.path.join(self.outputDir, ticName, '*.fits'))
        print(fitsList)
        return fitsList
        
    def get_cadence_data(self, fitsFile):
        fitsData = fits.open(fitsFile)
        cadenceData = fitsData[1].data # get the binary table
        binHeader = fitsData[1].header # for the binary table keywords
        priHeader = fitsData[0].header # for the primary table keywords
        
        pixelData = {}

        # read important keywords
        # the reference pixel coordinates are the low corner of the central pixel
        pixelData["referenceCol"] = binHeader["1CRPX4"] # reference pixel along axis 1
        pixelData["referenceRow"] = binHeader["2CRPX4"] # reference pixel along axis 1
        pixelData["referenceRa"] = binHeader["1CRVL4"] # reference pixel ra in degrees
        pixelData["referenceDec"] = binHeader["2CRVL4"] # reference pixel dec in degrees
        pixelData["cornerCol"] = binHeader["1CRV4P"] # lowest corner column
        pixelData["cornerRow"] = binHeader["2CRV4P"] # lowest corner row
        pixelData["sector"] = priHeader["SECTOR"] # FFI sector
        pixelData["camera"] = priHeader["CAMERA"] # FFI camera
        pixelData["ccd"] = priHeader["CCD"] # FFI ccd

        pixelData["rawTime"] = cadenceData["TIME"]
        pixelData["ffiBarycentricCorrection"] = cadenceData["TIMECORR"]
        pixelData["flux"] = np.flip(cadenceData["FLUX"], axis=1)
        pixelData["fluxErr"] = np.flip(cadenceData["FLUX_ERR"], axis=1)
        pixelData["background"] = np.flip(cadenceData["FLUX_BKG"], axis=1)
        pixelData["backgroundErr"] = np.flip(cadenceData["FLUX_BKG_ERR"], axis=1)
        pixelData["quality"] = cadenceData["QUALITY"]
        
        image_jd = cadenceData["TIME"]
        cadence0 = priHeader["FFIINDEX"]
        t0 = image_jd[0]

        dt = np.nanmedian(np.diff(image_jd))
        
#        if (image_jd[1] - image_jd[0]) > 0.01:
#            # 30 min cadence
#            dt = 48
#        elif (image_jd[1] - image_jd[0]) > 0.005:
#            # 10 min cadence
#            dt = 24*6
#        else:
#            # 2 min cadence
#            dt = 24*30

        pixelData["cadenceNumber"] = cadence0 + np.round((image_jd - t0)/dt).astype(int)

        # perform the barycentric correction
        # Tess cut time is barycentric correccted to the center of the source FFI
        # so compute the spacecraft clock time by subtracting the supplied
        # barycentric correction, and them computing the barycentric correction
        # for the target RA and Dec.  We use the reference pixel RA and Dec
        tessToJulianOffset = 2457000
        spacecraftTime = pixelData["rawTime"] - pixelData["ffiBarycentricCorrection"]
        pixelData["barycentricCorrection"], pixelData["validBarycentricCorrection"] = self.baryCorrector.computeCorrection(spacecraftTime + tessToJulianOffset,
                                                 pixelData["referenceRa"], pixelData["referenceDec"])
        pixelData["time"] = spacecraftTime + pixelData["barycentricCorrection"]
        if np.any(pixelData["validBarycentricCorrection"] == False):
            print("Some cadences have an invalid barycentric correction")
            
    #    print("time: " + str([np.min(pixelData["time"]), np.max(pixelData["time"])]))
        return pixelData
        
    def find_transit_times(self, pixelData, planetData):
        dt = np.min(np.diff(pixelData["time"]))
        nTransit = np.unique(np.round((pixelData["time"] - planetData["epoch"])/planetData["period"]).astype(int))
        transitTimes = planetData["epoch"] + planetData["period"] * nTransit
#        print("nTransit = " + str(nTransit) + ", transitTimes = " + str(transitTimes))
        transitIndex = np.array([np.abs(pixelData["time"] - t).argmin() for t in transitTimes])
        bufferRatio = 0.5
        flagGaps = np.abs(pixelData["time"][transitIndex] - transitTimes) > bufferRatio*dt
#        for i in np.nonzero(flagGaps)[0]:
#            print("large cadence difference: " + str(pixelData["time"][transitIndex][i] - transitTimes[i]))
        transitTimes = transitTimes[~flagGaps]
        transitIndex = transitIndex[~flagGaps]
        return transitTimes, transitIndex

    def find_transits(self, pixelData, planetData, allowedBadCadences = None):
        if allowedBadCadences is None:
            allowedBadCadences = self.allowedBadCadences
            
        transitTimes, transitIndex = self.find_transit_times(pixelData, planetData)
#        print("transitTimes = " + str(transitTimes))
        if len(transitTimes) == 0:
            return [],[],[],{}
          
        durationDays = planetData["durationHours"]/24;
        transitAverageDurationDays = 0.9*durationDays/2;
        # Center of the out transit is n cadences + one duration away from the center of the transit
        dt = np.min(np.diff(pixelData["time"])) # days
        outTransitBuffer = dt + durationDays
    #    print("allowedBadCadences = " + str(allowedBadCadences))
    #    print("durationDays = " + str(durationDays))
    #    print("outTransitBuffer = " + str(outTransitBuffer))
    #    print("transitAverageDurationDays = " + str(transitAverageDurationDays))
        expectedInTransitLength = np.floor(2*transitAverageDurationDays/dt)
#        print("expected number in transit = ", str(expectedInTransitLength))
        inTransitIndices = []
        outTransitIndices = []
        nBadCadences = []
        DiffImageDataList = []
#        print("raw transitIndex = " + str(transitIndex))
        for i in transitIndex:
            thisTransitInIndices = np.nonzero(
                (np.abs(pixelData["time"][i] - pixelData["time"]) < transitAverageDurationDays))[0]
            thisTransitOutIndices = np.nonzero(
                (np.abs(pixelData["time"][i] - pixelData["time"]) > (outTransitBuffer - transitAverageDurationDays))
                & (np.abs(pixelData["time"][i] - pixelData["time"]) < (outTransitBuffer + transitAverageDurationDays)))[0]
            thisTransitBadCadences = np.sum(pixelData["quality"][thisTransitInIndices] != 0) + np.sum(pixelData["quality"][thisTransitOutIndices] != 0)

            # check if we have indices covering the transits
            acceptableTransitLength = np.floor(self.allowedInTransitLossFraction*expectedInTransitLength)
            if len(thisTransitInIndices) < acceptableTransitLength:
                print("not enough in transit indices: " + str([len(thisTransitInIndices), acceptableTransitLength]))
                print("adding " + str(acceptableTransitLength - len(thisTransitInIndices)) + " to thisTransitBadCadences")
                thisTransitBadCadences += acceptableTransitLength - len(thisTransitInIndices)
            if len(thisTransitOutIndices) < 2*acceptableTransitLength:
                print("not enough out transit indices: " + str([len(thisTransitOutIndices), 2*acceptableTransitLength]))
                print("adding " + str(2*acceptableTransitLength - len(thisTransitOutIndices)) + " to thisTransitBadCadences")
                thisTransitBadCadences += 2*acceptableTransitLength - len(thisTransitOutIndices)

    #        print(sum(pixelData["quality"][thisTransitInIndices] > 0) + sum(pixelData["quality"][thisTransitOutIndices] > 0))
    #        print("transit " + str(i) + ":")
    #        print([len(thisTransitInIndices), expectedInTransitLength])
    #        print([len(thisTransitOutIndices), 2*expectedInTransitLength])
    #        if np.any(pixelData["quality"][thisTransitInIndices] > 0):
    #            print("in transit bad quality flags: " + str(pixelData["quality"][thisTransitInIndices]))
    #        if np.any(pixelData["quality"][thisTransitOutIndices] > 0):
    #            print("out transit bad quality flags: " + str(pixelData["quality"][thisTransitOutIndices]))
    
#            print(str(i) + " thisTransitBadCadences = " + str(thisTransitBadCadences))
            thisTransitInIndices = thisTransitInIndices[pixelData["quality"][thisTransitInIndices] == 0].tolist()
            thisTransitOutIndices = thisTransitOutIndices[pixelData["quality"][thisTransitOutIndices] == 0].tolist()

#            if (len(thisTransitInIndices) < expectedInTransitLength) | (len(thisTransitOutIndices) < 2*expectedInTransitLength):
##                print("not enough in/out transit indices: " + str([len(thisTransitInIndices), len(thisTransitOutIndices), expectedInTransitLength]))
##                print("not enough in/out transit indices, thisTransitBadCadences = " + str(thisTransitInIndices))
#                inTransitIndices.append([])
#                outTransitIndices.append([])
#                nBadCadences.append(allowedBadCadences+1)
##                print("building nBadCadences = " + str(nBadCadences))
#                DiffImageDataList.append([])
#                continue
            DiffImageDataList.append(self.make_difference_image(pixelData, thisTransitInIndices, thisTransitOutIndices))

            inTransitIndices.append(thisTransitInIndices)
            outTransitIndices.append(thisTransitOutIndices)
#            print("thisTransitBadCadences = " + str(thisTransitBadCadences))
            nBadCadences.append(thisTransitBadCadences)
#            print("building nBadCadences = " + str(nBadCadences))

        if len(nBadCadences) == 0:
            nBadCadences = [0]
        alert=False
#        print("nBadCadences = " + str(nBadCadences))
        if np.min(nBadCadences) > allowedBadCadences:
            print("No good transits based on %i allowed bad cadences; using transit with %i bad cadences." % (allowedBadCadences, np.min(nBadCadences)))
            alert=True
        goodTransits = (nBadCadences <= np.max([allowedBadCadences, np.min(nBadCadences)]))
#        print(nBadCadences)
#        print(goodTransits)
        
        diffImageData = {}
        nTransitImages = 0
        if len(DiffImageDataList) > 0:
            diffImageData["diffImage"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["diffImageSigma"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["diffSNRImage"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["meanInTransit"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["meanInTransitSigma"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["meanOutTransit"] = np.zeros(pixelData["flux"][0,:,:].shape)
            diffImageData["meanOutTransitSigma"] = np.zeros(pixelData["flux"][0,:,:].shape)
            for i in range(len(DiffImageDataList)):
#                print([i, goodTransits[i]])
                if goodTransits[i]:
#                    print("adding transit")
#                    if len(DiffImageDataList[i]) == 0:
#                        print(DiffImageDataList[i])
                    diffImageData["diffImage"] += DiffImageDataList[i]["diffImage"]
                    diffImageData["diffImageSigma"] += DiffImageDataList[i]["diffImageSigma"]**2
                    diffImageData["meanInTransit"] += DiffImageDataList[i]["meanInTransit"]
                    diffImageData["meanInTransitSigma"] += DiffImageDataList[i]["meanInTransitSigma"]**2
                    diffImageData["meanOutTransit"] += DiffImageDataList[i]["meanOutTransit"]
                    diffImageData["meanOutTransitSigma"] += DiffImageDataList[i]["meanOutTransitSigma"]**2
                    nTransitImages += 1
            if nTransitImages > 0:
                diffImageData["diffImage"] /= nTransitImages
                diffImageData["diffImageSigma"] = np.sqrt(diffImageData["diffImageSigma"])/nTransitImages
                diffImageData["meanInTransit"] /= nTransitImages
                diffImageData["meanInTransitSigma"] = np.sqrt(diffImageData["meanInTransitSigma"])/nTransitImages
                diffImageData["meanOutTransit"] /= nTransitImages
                diffImageData["meanOutTransitSigma"] = np.sqrt(diffImageData["meanOutTransitSigma"])/nTransitImages
                diffImageData["diffSNRImage"] = diffImageData["diffImage"]/diffImageData["diffImageSigma"]
            else:
                print("no good transits to sum into average images!!!")
        
        inTransitIndices = np.unique(sum(np.array(inTransitIndices, dtype=object)[goodTransits].tolist(), []))
        outTransitIndices = np.unique(sum(np.array(outTransitIndices, dtype=object)[goodTransits].tolist(), []))
#        print("final inTransitIndices = " + str(inTransitIndices))
#        print("final outTransitIndices = " + str(outTransitIndices))
        planetData["badCadenceAlert"] = alert
        
        return inTransitIndices, outTransitIndices, transitIndex, diffImageData

    def make_difference_image(self, pixelData, inTransitIndices, outTransitIndices):
        meanInTransit = np.nanmean(pixelData["flux"][inTransitIndices,::-1,:], axis=0)
        meanInTransitSigma = np.sqrt(np.sum(pixelData["fluxErr"][inTransitIndices,::-1,:]**2, axis=0)/len(inTransitIndices))
        meanOutTransit = np.nanmean(pixelData["flux"][outTransitIndices,::-1,:], axis=0)
        meanOutTransitSigma = np.sqrt(np.sum(pixelData["fluxErr"][outTransitIndices,::-1,:]**2, axis=0)/len(outTransitIndices))
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
        
    def make_stellar_scene(self, pixelData):
        catalogData = {}
        
        # compute mjd for the Gaia epoch J2015.5 = 2015-07-02T21:00:00
        t = Time("2015-07-02T21:00:00", format='isot', scale='utc')
        bjdJ2015p5 = t.jd - 2457000
        bjd = np.mean(pixelData["time"])
        mas2deg = 1/(3600*1000)
        dt = (bjd - bjdJ2015p5)/365
    #    print("dt = " + str(dt) + " years")

        searchRadius = (np.linalg.norm([pixelData["flux"].shape[1], pixelData["flux"].shape[2]]))*21/3600/2 # assumes 21 arcsec pixels
#        print("making catalog for " + str([self.ticData["raDegrees"], self.ticData["decDegrees"]]))
        ticCatalog = get_tic(self.ticData["raDegrees"], self.ticData["decDegrees"], searchRadius)
#        print(list(ticCatalog))
#        print(len(ticCatalog))
        dRa = mas2deg*dt*ticCatalog["pmRA"]/np.cos(ticCatalog["Dec_orig"]*np.pi/180)
        dRa[np.isnan(dRa)] = 0
        dDec = mas2deg*dt*ticCatalog["pmDEC"]
        dDec[np.isnan(dDec)] = 0
#        print("mean dRa in arcsec = " + str(3600*np.mean(dRa)))
#        print("mean dDec in arcsec = " + str(3600*np.mean(dDec)))
        ticCatalog["correctedRa"] = ticCatalog["RA_orig"] + dRa
        ticCatalog["correctedDec"] =  ticCatalog["Dec_orig"] + dDec

        targetIndex = np.where(np.array(ticCatalog["ID"]).astype(int)==self.ticData["id"])[0][0]
        if targetIndex > 0:
            targetTable = astropy.table.Table(ticCatalog[targetIndex])
            ticCatalog.remove_row(targetIndex)
            ticCatalog = astropy.table.vstack([targetTable, ticCatalog])
            targetIndex = 0

        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
                refColPix, refRowPix, scinfo = trdp.tess_stars2px_function_entry(
                    self.ticData["id"], pixelData["referenceRa"], pixelData["referenceDec"], aberrate=True, trySector=pixelData["sector"])
        onPix = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
        outID = outID[onPix]
#        print("len(outID) = " + str(len(outID)))
        if len(outID) == 0:
            print("no stars on pixels")
            return []
        catalogData["refColPix"] = refColPix[onPix]
        catalogData["refRowPix"] = refRowPix[onPix]
#        print("refpix = " + str([catalogData["refColPix"], catalogData["refRowPix"]]))

        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
        targetColPix, targetRowPix, scinfo = trdp.tess_stars2px_function_entry(
            self.ticData["id"], ticCatalog["correctedRa"][targetIndex], ticCatalog["correctedDec"][targetIndex], aberrate=True, trySector=pixelData["sector"], scInfo=scinfo)
        onPix = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
        outID = outID[onPix]
        if len(outID) == 0:
            print("target not on pixels")
            return []
        catalogData["targetColPix"] = targetColPix[onPix]
        catalogData["targetRowPix"] = targetRowPix[onPix]
    #    print([catalogData["targetColPix"], catalogData["targetRowPix"]])

        ticID, outEclipLong, outEclipLat, outSec, outCam, outCcd, ticColPix, ticRowPix, scinfo \
            = trdp.tess_stars2px_function_entry(
                ticCatalog["ID"], ticCatalog["correctedRa"], ticCatalog["correctedDec"],
                aberrate=True, trySector=pixelData["sector"], scInfo=scinfo)
        theseStars = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
#        print("theseStars = " + str(theseStars))
        correctedRa = ticCatalog["correctedRa"]
        correctedDec = ticCatalog["correctedDec"]
        deltaRa = 3600*np.cos(ticCatalog["correctedDec"][targetIndex]*np.pi/180)*(correctedRa - ticCatalog["correctedRa"][targetIndex])
        deltaDec = 3600*(correctedDec - ticCatalog["correctedDec"][targetIndex])
        separation = np.sqrt(deltaRa**2 + deltaDec**2)

        catalogData["ticID"] = ticID[theseStars]
        catalogData["ticColPix"] = ticColPix[theseStars]
        catalogData["ticRowPix"] = ticRowPix[theseStars]
        catalogData["correctedRa"] = correctedRa[np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["correctedDec"] = correctedDec[np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["deltaRa"] = deltaRa[np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["deltaDec"] = deltaDec[np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["separation"] = separation[np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["ticMag"] = ticCatalog["Tmag"][np.isin(ticCatalog["ID"], catalogData["ticID"])]
        catalogData["ticFlux"] = mag2flux(catalogData["ticMag"])
#        print("ticFlux = " + str(catalogData["ticFlux"]))
        catalogData["ticFluxNorm"] = np.sqrt(0.999*catalogData["ticFlux"]/np.max(catalogData["ticFlux"]) + 0.001)
    #    print([len(catalogData["ticID"]), len(catalogData["ticMag"])])

    #    extent = (pixelData["cornerCol"], pixelData["cornerCol"] + 20, pixelData["cornerRow"], pixelData["cornerRow"] + 20)
    #    extent = (pixelData["cornerRow"]-0.5, pixelData["cornerRow"]-0.5 + 20, pixelData["cornerCol"]-0.5, pixelData["cornerCol"]-0.5 + 20)
        catalogData["extent"] = (pixelData["cornerCol"] - 0.5, pixelData["cornerCol"] + pixelData["flux"].shape[1] - 0.5,
            pixelData["cornerRow"] - 0.5, pixelData["cornerRow"] + pixelData["flux"].shape[2] - 0.5)
    #    print(pixelData["flux"].shape)
    #    print(catalogData["extent"])
        catalogData["dRow"] = catalogData["refRowPix"] - (pixelData["referenceRow"] + catalogData["extent"][2] - 0.5)
        catalogData["dCol"] = catalogData["refColPix"] - (pixelData["referenceCol"] + catalogData["extent"][0] - 0.5)

        closeupSize = 5
        closeupOffset = np.round(float(closeupSize)/2)
#        catalogData["extentClose"] = (pixelData["cornerCol"] + 8 - 0.5, pixelData["cornerCol"] + 8 + closeupSize - 0.5,
#            pixelData["cornerRow"] + 8 - 0.5, pixelData["cornerRow"]  + 8 + closeupSize - 0.5)
#        print("lengh of catalogData = " + str(len(catalogData)))
#        print("lengh of catalogData[targetColPix] = " + str(len(catalogData["targetColPix"])))
        centerColPix = np.round(catalogData["targetColPix"][0])
        centerRowPix = np.round(catalogData["targetRowPix"][0])
        catalogData["extentClose"] = (centerColPix - closeupOffset - 0.5,
                                        centerColPix - closeupOffset + closeupSize - 0.5,
                                        centerRowPix - closeupOffset - 0.5,
                                        centerRowPix - closeupOffset + closeupSize - 0.5)

        f = open(os.path.join(self.outputDir, self.ticName, "ticKey_" + self.ticName + ".txt"), 'w')
        f.write("# index, TIC ID, TMag, separation (arcsec), ra, dec, deltaRa (arcsec), deltaDec (arcsec)\n")
        for s, id in enumerate(catalogData["ticID"]):
    #        if (catalogData["ticMag"][s]-catalogData["ticMag"][0] < dMagThreshold):
            f.write(str(s) + ", " + str(id) + ", " + str(catalogData["ticMag"][s]) + ", " + str(np.round(catalogData["separation"][s], 3)) + ", " + str(np.round(catalogData["correctedRa"][s], 3)) + ", " + str(np.round(catalogData["correctedDec"][s], 3)) + ", " + str(np.round(catalogData["deltaRa"][s], 3)) + ", " + str(np.round(catalogData["deltaDec"][s], 3)) + "\n")
        f.close()

        return catalogData
        
    def draw_pix_catalog(self, pixArray, catalogData, extent, hiliteStar=None, ax=None, close=False, dMagThreshold = None, annotate=False, printMags=False, magColorBar=False, pixColorBar=True, pixColorBarLabel=False, filterStars=False, starsToAnnotate = [], targetID=None, fs=18, ss=400):
    
        dCol = catalogData["dCol"]
        dRow = catalogData["dRow"]
        if dMagThreshold is None:
            dMagThreshold = self.dMagThreshold
        if ax is None:
            ax = plt.gca()
        if targetID is None:
            targetIndex = 0
        else:
            targetIndex = ((ticCatalog["ID"]).astype(int)==targetID)[0]
        if close:
            cSize = np.floor(extent[1] - extent[0]).astype(int) # assume square
            if (pixArray.shape[0] > cSize) | (pixArray.shape[1] > cSize):
                cCol = np.floor(extent[0]-catalogData["extent"][0]).astype(int)
                cRow = np.floor(extent[2]-catalogData["extent"][2]).astype(int)
                # slice operations pick out rows and columns, not x and y
                pixArray=pixArray[cRow:cRow + cSize,cCol:cCol + cSize]
        im = ax.imshow(pixArray, cmap='jet', origin="lower", extent=extent)
        if pixColorBar:
            cbh = plt.colorbar(im, ax=ax)
            cbh.ax.tick_params(labelsize=fs-2)
        if pixColorBarLabel:
            cbh.ax.set_ylabel("Pixel Flux [e$^-$/sec]", fontsize=fs-2)
        if not close:
            ax.plot([catalogData["extent"][0], catalogData["extent"][1]], [catalogData["targetRowPix"] - dRow,catalogData["targetRowPix"] - dRow], 'r', alpha = 0.6)
            ax.plot([catalogData["targetColPix"] - dCol,catalogData["targetColPix"] - dCol], [catalogData["extent"][2], catalogData["extent"][3]], 'r', alpha = 0.6)
        ax.plot(catalogData["targetColPix"] - dCol, catalogData["targetRowPix"] - dRow, 'm*', zorder=100, ms=fs-2)
        if ss > 0:
            targetMag = catalogData["ticMag"][targetIndex]
            if filterStars:
                idx = (catalogData["ticMag"]-targetMag) < dMagThreshold
            else:
                idx = range(len(catalogData['ticID']))
            star_gs = ax.scatter(catalogData["ticColPix"][idx]  - dCol, catalogData["ticRowPix"][idx] - dRow, cmap='BuGn',
                c=catalogData["ticMag"][idx], s=ss*catalogData["ticFluxNorm"][idx]**2, edgecolors="w", linewidths=0.5, alpha=1, zorder=10)
            if hiliteStar is not None:
                if type(hiliteStar) is not list:
                    hiliteStar = [hiliteStar]
                for hs in hiliteStar:
                    # set minimum size for the hilight icon
                    hilightSize = np.max([catalogData["ticFluxNorm"][hs], 0.085])
                    plt.scatter(catalogData["ticColPix"][hs]-dCol, catalogData["ticRowPix"][hs]-dRow, s=ss*5*hilightSize**2, marker="s",
                            color="y", edgecolor="k", linewidths=0.5, alpha = 0.6, zorder=1)
            if magColorBar:
                cbh2 = plt.colorbar(star_gs, ax=ax)
                cbh2.ax.set_ylabel('T mag', fontsize=fs-2)
                cbh2.ax.tick_params(labelsize=fs-2)
            if annotate:
                bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                pscale = bbox.width * plt.gcf().dpi
                for s in range(len(catalogData['ticID'])):
                    px = catalogData["ticColPix"][s] - dCol
                    py = catalogData["ticRowPix"][s] - dRow
                    ticMag = catalogData["ticMag"][s]
                    if (((s in starsToAnnotate) | (ticMag-targetMag < dMagThreshold)) & (px >= extent[0]) & (px <= extent[1]) & (py > extent[2]) & (py < extent[3])):
                        if printMags:
                            starStr = str(s) + ", " + str(np.round(ticMag,1))
                        else:
                            starStr = str(s)
                        ax.text(px, py + 1*20/pscale, starStr, color="w", fontsize = fs-2, path_effects=[pe.withStroke(linewidth=1,foreground='black')], zorder=100)
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def draw_difference_image(self, diffImageData, pixelData, planetData, catalogData, dMagThreshold = None):
        if dMagThreshold is None:
            dMagThreshold = self.dMagThreshold

        if planetData["badCadenceAlert"]:
            alertText = ", no good transits!!!"
            alertColor = "r"
        else:
            alertText = ""
            alertColor = "k"
        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffImage"], catalogData, extent=catalogData["extent"], ax=ax, dMagThreshold = dMagThreshold)
        plt.title("diff image" + alertText, color=alertColor);
        plt.savefig(os.path.join(self.outputDir, self.ticName, "diffImage_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffImage"], catalogData, extent=catalogData["extentClose"], ax=ax, close=True)
        plt.title("diff image close" + alertText, color=alertColor);
        plt.savefig(os.path.join(self.outputDir, self.ticName, "diffImageClose_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffSNRImage"], catalogData, extent=catalogData["extent"], ax=ax, dMagThreshold = dMagThreshold)
        plt.title("SNR diff image" + alertText, color=alertColor);
        plt.savefig(os.path.join(self.outputDir, self.ticName, "diffImageSNR_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffSNRImage"], catalogData, extent=catalogData["extentClose"], ax=ax, close=True)
        plt.title("SNR diff image close" + alertText, color=alertColor);
        plt.savefig(os.path.join(self.outputDir, self.ticName, "diffImageSNRClose_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, extent=catalogData["extent"], ax=ax, magColorBar=True, dMagThreshold = dMagThreshold)
        plt.title("Direct image");
        plt.savefig(os.path.join(self.outputDir, self.ticName, "directImage_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, extent=catalogData["extent"], ax=ax, annotate=True, magColorBar=True, dMagThreshold = dMagThreshold)
        plt.title("Direct image");
        plt.savefig(os.path.join(self.outputDir, self.ticName, "directImageAnnotated_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, extent=catalogData["extentClose"], ax=ax, close=True)
        plt.title("Direct image close");
        plt.savefig(os.path.join(self.outputDir, self.ticName, "directImageClose_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

    def draw_lc_transits(self, pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex, apCenter = [10,10], saveFigure=True):
    #    apFlux = pixelData["flux"][:,8:13,8:13]
        apFlux = pixelData["flux"][:,apCenter[0]-2:apCenter[0]+3,apCenter[1]-2:apCenter[1]+3]
        lc = np.sum(np.sum(apFlux,2),1)
        plt.figure(figsize=(15, 5));
        plt.plot(pixelData["time"][pixelData["quality"]==0], lc[pixelData["quality"]==0], label="flux")
        plt.plot(pixelData["time"][pixelData["inOtherTransit"]==1], lc[pixelData["inOtherTransit"]==1], 'y+', ms=20, alpha = 0.6, label="in other transit")
        plt.plot(pixelData["time"][pixelData["quality"]>0], lc[pixelData["quality"]>0], 'rx', ms=10, label="quality problems")
        plt.plot(pixelData["time"][inTransitIndices], lc[inTransitIndices], 'd', ms=10, alpha = 0.6, label="in transit")
        plt.plot(pixelData["time"][outTransitIndices], lc[outTransitIndices], 'o', ms=10, alpha = 0.5, label="out of transit")
        plt.plot(pixelData["time"][transitIndex], lc[transitIndex], 'b*', ms=10, alpha = 1, label="transit center", zorder=100)
        plt.legend()
        plt.xlabel("time");
        plt.ylabel("flux (e-/sec)");
        
        if saveFigure:
            plt.savefig(os.path.join(self.outputDir, self.ticName, "lcTransits_planet" + str(planetData["planetIndex"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf"),bbox_inches='tight')

