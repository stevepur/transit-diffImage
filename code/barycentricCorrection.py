
import spiceypy as spice
from astropy.time import Time
import numpy as np
import array

class barycentricCorrection:
    def __init__(self, spiceFileLocation = ".", tlsFile = None, ephemFile = None, planetFile = None, spacecraftCode='-95'):
        self.spiceFileLocation = spiceFileLocation
    
        # code -95 is TESS, though it's called Mgs Simulation
        self.spacecraftCode = spacecraftCode
        
        if (tlsFile == None) & (spacecraftCode == '-95'):
            self.tlsFile = 'tess2018338154046-41240_naif0012.tls'
        else:
            if tlsFile == None:
                print("if we're not TESS we need a specified tls file!!")
                raise
            self.tlsFile = tlsFile
        if (planetFile == None) & (spacecraftCode == '-95'):
#            self.planetFile = 'tess2018338154429-41241_de430.bsp'
            self.planetFile = 'de432s.bsp'
        else:
            if planetFile == None:
                print("if we're not TESS we need a specified planet ephemeris (e.g. de430) file!!")
                raise
            self.planetFile = planetFile
        if (ephemFile == None) & (spacecraftCode == '-95'):
            self.ephemFile = 'TESS_merge_ephem.bsp'
        else:
            if ephemFile == None:
                print("if we're not TESS we need a specified ephemeris file!!")
                raise
            self.bspFile = ephemFile
            
        spice.furnsh(self.spiceFileLocation + "/" + self.tlsFile)
        spice.furnsh(self.spiceFileLocation + "/" + self.planetFile)
        spice.furnsh(self.spiceFileLocation + "/" + self.ephemFile)

    def computeCorrection(self, spacecraftJulianTime, ra, dec):

        julianTimes = Time(spacecraftJulianTime, format='jd', scale='utc')
        times = spice.str2et(julianTimes.iso)
        
        # in case the input is just a float with no len
        if not hasattr(times, "__len__"):
            times = [times]
            
        state = np.zeros((len(times),6))
        valid = np.zeros(len(times), dtype = bool)
        # we have to get the state for each time individually, becausee if there
        # is a gap in the ephemeris spkezr called with a vector of times will
        # fail for all times after that gap
        for i in range(len(times)):
            try:
                # spice.spkezr will raise an exception if the time is not covered by
                # the loaded ephemeris
                state[i,:], lightTimes = spice.spkezr(self.spacecraftCode, times[i], 'J2000', 'NONE', 'SSB')
                valid[i] = True
            except:
                state[i,:] = np.zeros(state.shape[1])
                valid[i] = False

        # from Kepler SOC code get_kepler_to_barycentric_offset.m
        # construct the unit vectors pointing towards RA and Dec in the coordinate system.  To do
        # this, we will convert RA and Dec into more conventional spherical coordinates, to wit:
        #
        #     theta = angle from +x-axis towards the +y-axis,
        #     phi = angle from the +z-axis towards the x-y plane.
        #
        # In this coordinate system, the transformation into a Cartesian unit vector is:
        #
        #    x = cos(theta)*sin(phi)
        #    y = sin(theta)*sin(phi)
        #    z = cos(phi)
        deg2rad = np.pi/180
        sec2day = 1/24/60/60
        theta = ra*deg2rad
        phi   = (90-dec)*deg2rad
        targetUnitVector = np.array([np.cos(theta)*np.sin(phi),  np.sin(theta)*np.sin(phi),  np.cos(phi)])
#        print(targetUnitVector.shape)
#        print(state[:,0:3].shape)

        barycentricCorrectionSeconds = targetUnitVector @ state[:,0:3].T / spice.clight()

        return barycentricCorrectionSeconds * sec2day, valid


