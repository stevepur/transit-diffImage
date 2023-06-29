"""Provides callable models of the Kepler Pixel Response Function (PRF)."""
from __future__ import division, print_function

import math

from astropy.io import fits as pyfits
import numpy as np
import scipy
import scipy.interpolate

#from ..utils import channel_to_module_output, plot_image


__all__ = ['tessPRF', 'SimpletessPRF']


class tessPRF(object):
    """
    Kepler's Pixel Response Function as designed by [1]_.

    This class provides the necessary interface to load Kepler PRF
    calibration files and to create a model that can be fit as a function
    of flux, center positions, width, and rotation angle.

    Attributes
    ----------
    channel : int
        KeplerTargetPixelFile.channel
    shape : (int, int)
        KeplerTargetPixelFile.shape[1:]
    column : int
        KeplerTargetPixelFile.column
    row : int
        KeplerTargetPixelFile.row

    Examples
    --------
    Objects from the tessPRF class are defined by a channel number, a pair of
    dimensions (the size of the image), and a reference coordinate (bottom left
    corner). In this example, we create a tessPRF object located at channel
    #44 with dimension equals 10 x 10, reference row and column coordinate
    equals (5, 5). After the object has been created, we may translate it to a
    given center coordinate. Additionally, we can specify total flux, pixel
    scales, and rotation around the object's center.

    >>> import math
    >>> import matplotlib.pyplot as plt
    >>> from lightkurve import tessPRF
    >>> kepprf = tessPRF(channel=44, shape=(10, 10), column=5, row=5) # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/fpc/prf/kplr13.4_2011265_prf.fits [Done]
    >>> prf = kepprf(flux=1000, center_col=10, center_row=10,
    ...              scale_row=0.7, scale_col=0.7, rotation_angle=math.pi/2) # doctest: +SKIP
    >>> plt.imshow(prf, origin='lower') # doctest: +SKIP

    References
    ----------
    .. [1] S. T. Bryson. The Kepler Pixel Response Function, 2010.
           <https://arxiv.org/abs/1001.0331>.
    """

    def __init__(self, shape, sector, camera, ccd, column, row, prfFileLocation = None):
        self.sector = sector
        self.shape = shape
        self.camera = camera
        self.ccd = ccd
        self.column = column
        self.row = row
        self.prfFileLocation = prfFileLocation

        self.col_coord, self.row_coord, self.interpolate, self.supersampled_prf = self._prepare_prf()

    def __call__(self, center_col, center_row, flux, scale_col, scale_row,
                 rotation_angle):
        return self.evaluate(center_col, center_row, flux,
                             scale_col, scale_row, rotation_angle)

    def evaluate(self, center_col, center_row, flux=1., scale_col=1., scale_row=1.,
                 rotation_angle=0.):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF
        scale_col, scale_row : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        prf_model : 2D array
            Two dimensional array representing the PRF values parametrized
            by flux, centroids, widths, and rotation.
        """
        cosa = math.cos(rotation_angle)
        sina = math.sin(rotation_angle)

        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row
        delta_col, delta_row = np.meshgrid(delta_col, delta_row)

        rot_row = delta_row * cosa - delta_col * sina
        rot_col = delta_row * sina + delta_col * cosa

        self.prf_model = flux * self.interpolate(rot_row.flatten() * scale_row,
                                                 rot_col.flatten() * scale_col,
                                                 grid=False).reshape(self.shape)
        return self.prf_model

    def gradient(self, center_col, center_row, flux=1., scale_col=1., scale_row=1.,
                 rotation_angle=0.):
        """
        This function returns the gradient of the tessPRF model with
        respect to center_col, center_row, flux, scale_col, scale_row,
        and rotation_angle.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF
        scale_col, scale_row : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        grad_prf : list
            Returns a list of arrays where the elements are the partial derivatives
            of the tessPRF model with respect to center_col, center_row, flux, scale_col,
            scale_row, and rotation_angle, respectively.
        """
        cosa = math.cos(rotation_angle)
        sina = math.sin(rotation_angle)

        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row
        delta_col, delta_row = np.meshgrid(delta_col, delta_row)

        rot_row = delta_row * cosa - delta_col * sina
        rot_col = delta_row * sina + delta_col * cosa

        # for a proof of the maths that follow, see the pdf attached
        # on pull request #198 in lightkurve GitHub repo.
        deriv_flux = self.interpolate(rot_row.flatten() * scale_row,
                                      rot_col.flatten() * scale_col,
                                      grid=False).reshape(self.shape)

        interp_dy = self.interpolate(rot_row.flatten() * scale_row,
                                     rot_col.flatten() * scale_col,
                                     grid=False, dy=1).reshape(self.shape)

        interp_dx = self.interpolate(rot_row.flatten() * scale_row,
                                     rot_col.flatten() * scale_col,
                                     grid=False, dx=1).reshape(self.shape)

        scale_row_times_interp_dx = scale_row * interp_dx
        scale_col_times_interp_dy = scale_col * interp_dy

        deriv_center_col = - flux * (cosa * scale_col_times_interp_dy - sina * scale_row_times_interp_dx)
        deriv_center_row = - flux * (sina * scale_col_times_interp_dy + cosa * scale_row_times_interp_dx)
        deriv_scale_row = flux * interp_dx * rot_row
        deriv_scale_col = flux * interp_dy * rot_col
        deriv_rotation_angle = flux * (interp_dy * scale_col * (delta_row * cosa - delta_col * sina)
                                       - interp_dx * scale_row * (delta_row * sina + delta_col * cosa))

        return [deriv_center_col, deriv_center_row, deriv_flux,
                deriv_scale_col, deriv_scale_row, deriv_rotation_angle]

    def _read_prf_calibration_file(self, path):
#        prf_cal_file = pyfits.open(path)
        prf_cal_file = pyfits.open(path)
        data = prf_cal_file[0].data
        # looks like these data below are the same for all prf calibration files
        crval1p = prf_cal_file[0].header['CRVAL1P']
        crval2p = prf_cal_file[0].header['CRVAL2P']
        cdelt1p = prf_cal_file[0].header['CDELT1P']
        cdelt2p = prf_cal_file[0].header['CDELT2P']
        prf_cal_file.close()

        return data, crval1p, crval2p, cdelt1p, cdelt2p

    def _prepare_prf(self):
        n_hdu = 25
        min_prf_weight = 1e-6
        # determine suitable PRF calibration file
#        print([self.sector, self.camera, self.ccd])
#
#
# old file naming
#
#
#        if self.sector < 4:
#            prfSector = 1
#            if (self.camera == 1) & (self.ccd == 1):
#                prfName = "tess2019107181900-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#            elif ((self.camera == 1) & (self.ccd > 1)):
#                prfName = "tess2019107181901-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#            elif ((self.camera == 2) & (self.ccd < 4)):
#                prfName = "tess2018243163600-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#            else:
#                prfName = "tess2018243163601-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#        else:
#            prfSector = 4
#            if (self.camera == 1) & (self.ccd == 1):
#                prfName = "tess2019107181900-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#            elif ((self.camera == 1) & (self.ccd > 1)) \
#                | (self.camera == 2) \
#                | ((self.camera == 3) & (self.ccd == 1)):
#                prfName = "tess2019107181901-prf-" + str(self.camera)+ "-"  + str(self.ccd)
#            else:
#                prfName = "tess2019107181902-prf-" + str(self.camera)+ "-"  + str(self.ccd)


        if self.sector < 4:
            prfSector = 1
            if ((self.camera == 1) | ((self.camera == 2) & (self.ccd < 4))):
                prfName = "tess2018243163600-prf-" + str(self.camera)+ "-"  + str(self.ccd)
            else:
                prfName = "tess2018243163601-prf-" + str(self.camera)+ "-"  + str(self.ccd)
        else:
            prfSector = 4
            if (self.camera == 1) & (self.ccd == 1):
                prfName = "tess2019107181900-prf-" + str(self.camera)+ "-"  + str(self.ccd)
            elif ((self.camera == 1) & (self.ccd > 1)) \
                | (self.camera == 2) \
                | ((self.camera == 3) & (self.ccd == 1)):
                prfName = "tess2019107181901-prf-" + str(self.camera)+ "-"  + str(self.ccd)
            else:
                prfName = "tess2019107181902-prf-" + str(self.camera)+ "-"  + str(self.ccd)

        if self.prfFileLocation is None:
            prfs_url_path = "http://archive.stsci.edu/missions/tess/models/prf_fitsfiles/"
            prfLocation = prfs_url_path + "start_s000" + str(prfSector) + "/cam" + str(self.camera) + "_ccd"  + str(self.ccd) + "/"
        else:
            prfLocation = self.prfFileLocation + "start_s000" + str(prfSector) + "/tess_prf_camera_" + str(self.camera) \
                + "/cam" + str(self.camera) + "_ccd"  + str(self.ccd) + "/"
        prfFileStub = prfLocation + prfName

        # read PRF images
        prfn = [0] * n_hdu
        crval1p = np.zeros(n_hdu, dtype='float32')
        crval2p = np.zeros(n_hdu, dtype='float32')
        cdelt1p = np.zeros(n_hdu, dtype='float32')
        cdelt2p = np.zeros(n_hdu, dtype='float32')

        rowPrfAtArray = [1, 513, 1025, 1536, 2048]
        colPrfAtArray = [45, 557, 1069, 1580, 2092]
        for i in range(len(rowPrfAtArray)):
            for j in range(len(colPrfAtArray)):
                prffile = prfFileStub + "-row" + "{:04d}".format(rowPrfAtArray[i]) + "-col" + "{:04d}".format(colPrfAtArray[j]) + ".fits"
#                print(prffile)
                prfn[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i] = self._read_prf_calibration_file(prffile)

        prfn = np.array(prfn, dtype=object)
        PRFcol = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
        PRFrow = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
        PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
        PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

        # interpolate the calibrated PRF shape to the target position
        rowdim, coldim = self.shape[0], self.shape[1]
        prf = np.zeros(np.shape(prfn[0]), dtype='float32')
        ref_column = self.column + .5 * coldim
        ref_row = self.row + .5 * rowdim

        for i in range(n_hdu):
            prf_weight = math.sqrt((ref_column - crval1p[i]) ** 2
                                   + (ref_row - crval2p[i]) ** 2)
            if prf_weight < min_prf_weight:
                prf_weight = min_prf_weight
            prf += prfn[i] / prf_weight

        prf /= (np.nansum(prf) * cdelt1p[0] * cdelt2p[0])

        # location of the data image centered on the PRF image (in PRF pixel units)
#        col_coord = np.arange(self.column + .5, self.column + coldim + .5)
#        row_coord = np.arange(self.row + .5, self.row + rowdim + .5)
        col_coord = np.arange(self.column+1, self.column + coldim+1)
        row_coord = np.arange(self.row+1, self.row + rowdim+1)
        # x-axis correspond to row-axis in scipy.RectBivariate
        # not to be confused with our convention, in which the
        # x-axis correspond to the column-axis
        interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, prf)

        return col_coord, row_coord, interpolate, prf

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
#        plot_image(pflux, title='Kepler PRF Model, Channel: {}'.format(self.channel),
#                   extent=(self.column, self.column + self.shape[1],
#                           self.row, self.row + self.shape[0]), **kwargs)


class SimpleTessPRF(tessPRF):
    """
    Simple model of tessPRF.

    This class provides identical functionality as in tessPRF, except that
    it is parametrized only by flux and center positions. The width scales
    and angle are fixed to 1.0 and 0, respectivelly.
    """

    def __call__(self, center_col, center_row, flux=1.):
        return self.evaluate(center_col, center_row, flux)

    def evaluate(self, center_col, center_row, flux=1.):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        flux : float
            Total integrated flux of the PRF
        center_col, center_row : float
            Column and row coordinates of the center

        Returns
        -------
        prf_model : 2D array
            Two dimensional array representing the PRF values parametrized
            by flux and centroids.
        """
        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row
        self.prf_model = flux * self.interpolate(delta_row, delta_col)

        return self.prf_model

    def gradient(self, center_col, center_row, flux):
        """
        This function returns the gradient of the SimpletessPRF model with
        respect to flux, center_col, and center_row.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF

        Returns
        -------
        grad_prf : list
            Returns a list of arrays where the elements are the derivative
            of the tessPRF model with respect to center_col, center_row,
            and flux, respectively.
        """
        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row

        deriv_flux = self.interpolate(delta_row, delta_col)
        deriv_center_col = - flux * self.interpolate(delta_row, delta_col, dy=1)
        deriv_center_row = - flux * self.interpolate(delta_row, delta_col, dx=1)

        return [deriv_center_col, deriv_center_row, deriv_flux]


