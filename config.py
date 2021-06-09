import astropy.units as u
import numpy as np
from astropy.io import fits

import matplotlib as mpl
from matplotlib import rc
from astropy.constants import c, h, k_B

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif'})#,'sans-serif':['Helvetica']})
rc('text', usetex=True)
# set tickmarks inwards
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# file_HC3N_10_9_TdV_mJy
# N2Hp_TdV = 'data/Per-emb-2-N2Hp-1-0_TdV.fits'
HC3N_10_9_Vlsr = 'data/Per-emb-2-HC3N_10-9_fit_Vc.fits'
HC3N_10_9_TdV = 'data/Per-emb-2-HC3N_10-9_TdV.fits'
N2Dp_TdV = 'data/Per-emb-2-N2Dp-1-0_TdV.fits'
N2Dp_Vlsr = 'data/Per-emb-2-N2Dp_1-0_fit_Vc.fits'
N2Dp_eVlsr = 'data/Per-emb-2-N2Dp_1-0_fit_eVc.fits'
N2Hp_TdV = 'data/Per-emb-2-N2Hp-1-0_TdV.fits'
#
ra_Per2 = 15 * (3 + (32 + 17.92/60.) / 60.) * u.deg
dec_Per2 = (30 + (49 + 48.03 / 60.) / 60.) * u.deg
distance = 300.
N2Dp_TdV_levels = 19e-3 * np.arange(5, 20, 3)
N2Hp_TdV_levels = 30e-3 * np.arange(5, 20, 3)
HC3N_10_9_TdV_levels = 8e-3 * np.arange(5, 20, 3)

Per2_inc = -43*u.deg
Per2_pa = 130*u.deg
Per2_v_lsr = 7.05*u.km/u.s

region_file = 'Streamer_North_v2.reg'

# CO contour levels to show outflow
file_12CO_blue = 'data/Per-emb-2-CO_2-1-TdV-blue.fits'
file_12CO_red = 'data/Per-emb-2-CO_2-1-TdV-red.fits'
CO_red_levs = 0.2 * np.arange(5, 55, 10)
CO_blue_levs = 0.2 * np.arange(5, 55, 10)
#
# Polygon coordinates added by hand from the region file
# The next line is to close the polygon
poly = np.array([53.079519, 30.83452, 53.078404, 30.836177, 53.076689, 30.83787, 53.075746, 30.837944,
                 53.075017, 30.837208, 53.075067, 30.835867, 53.076296, 30.833681, 53.07735, 30.831875,
                 53.077525, 30.829164, 53.078662, 30.827808, 53.079905, 30.82885, 53.080033, 30.832016])
ra_poly = np.append(np.reshape(poly, [-1, 2])[:, 0], poly[0])*u.deg
dec_poly = np.append(np.reshape(poly, [-1, 2])[:, 1], poly[1])*u.deg


def per_emb_2_get_vc_proj_r(file_in=N2Dp_Vlsr, pa_angle=0*u.deg, inclination=0*u.deg):
    """
    Returns the centroid velocity and projected separation in the sky of the
    centroid velocity from Per-emb-2.
    r_proj is in u.au and V_los is in u.km/u.s
    :return: r_proj, V_los
    """
    from astropy.wcs import WCS
    from astropy.io import fits
    import velocity_tools.coordinate_offsets as c_offset
    # load region file and WCS structures
    wcs_Vc = WCS(file_in)
    #
    hd_Vc = fits.getheader(file_in)
    results = c_offset.generate_offsets(hd_Vc, ra_Per2, dec_Per2, 
        pa_angle=pa_angle, inclination=inclination)
    rad_au = (results.lon * distance*u.pc).to(u.au, equivalencies=u.dimensionless_angles())
    #
    Vc = fits.getdata(file_in)
    gd = np.isfinite(Vc) == 1
    v_los = Vc[gd]*u.km/u.s
    r_proj = rad_au[gd]
    return r_proj, v_los

def pb_interferometer(freq_obs, telescope='NOEMA'):
    """
    Primary beam diameter for SMA at the observed frequency.
        PB = 48.0 * (231.0*u.GHz) / freq_obs

    :param freq_obs: is the observed frequency in GHz.
    :return: The primary beam FWHM in arcsec
    """
    if telescope == 'NOEMA':
        return (64.1 * u.arcsec * 72.78382 * u.GHz / freq_obs).decompose()
    elif telescope == 'SMA':
        return (48.0 * u.arcsec * 231 * u.GHz / freq_obs).decompose()
    elif telescope == 'ALMA':
        return (21.0 * u.arcsec * 300 * u.GHz / freq_obs).decompose()
    elif telescope == 'ACA':
        return (35.0 * u.arcsec * 300 * u.GHz / freq_obs).decompose()
    elif telescope == 'VLA' or telescope == 'EVLA' or telescope == 'JVLA':
        return (45 * 60.0 * u.arcsec * u.GHz / freq_obs).decompose()
    else:
        print("Telescope still not included")
        return np.nan * u.arcsec

def setup_plot_noema(fig_i, label_col='black', star_col='red'):
    """
    Setup of NOEMA plots, since they will show all the same format.
    """
    fig_i.set_system_latex(True)
    fig_i.ticks.set_color(label_col)
    fig_i.recenter(53.075, 30.8299, radius=45 * (u.arcsec).to(u.deg))
    fig_i.set_nan_color('0.9')
    fig_i.add_beam(color=label_col)
    #
    ang_size = (5e3 / distance) * u.arcsec
    fig_i.add_scalebar(ang_size, label='5,000 au', color=label_col)
    fig_i.show_markers(ra_Per2.value, dec_Per2.value, marker='*', s=60, layer='star',
                       edgecolor=star_col, facecolor=label_col, zorder=31)
    fig_i.tick_labels.set_xformat('hh:mm:ss')
    fig_i.tick_labels.set_yformat('dd:mm:ss')
    fig_i.ticks.set_length(7)
    fig_i.axis_labels.set_xtext(r'Right Ascension (J2000)')
    fig_i.axis_labels.set_ytext(r'Declination (J2000)')
    return

def setup_plot_30m(fig_i, label_col='black', star_col='red'):
    """
    Setup of IRAM 30m plots, since they will show all the same format.
    """
    fig_i.set_system_latex(True)
    fig_i.ticks.set_color(label_col)
    # fig_i.recenter(53.075, 30.8299, radius=45 * (u.arcsec).to(u.deg))
    fig_i.set_nan_color('0.9')
    fig_i.add_beam(color=label_col)
    #
    ang_size = (5e3 / distance) * u.arcsec
    fig_i.add_scalebar(ang_size, label='5,000 au', color=label_col)
    fig_i.show_markers(ra_Per2.value, dec_Per2.value, marker='*', s=60, layer='star',
                       edgecolor=star_col, facecolor=label_col, zorder=31)
    fig_i.tick_labels.set_xformat('hh:mm:ss')
    fig_i.tick_labels.set_yformat('dd:mm')
    fig_i.ticks.set_length(7)
    fig_i.axis_labels.set_xtext(r'Right Ascension (J2000)')
    fig_i.axis_labels.set_ytext(r'Declination (J2000)')
    return

def convert_into_mili(file_name):
    """
    It converts a file into one rescaled by 1e3.
    This is useful to convert between Jy -> mJy or m/s into km/s
    for plotting purposes (e.g. to use with aplpy).

    Usage:
    fig = aplpy.FITSFigure(convert_into_mili(file_in_Jy), figsize=(4,4))
    fig.show_colorscale(vmin=0, vmax=160, cmap='inferno')
    fig.add_colorbar()
    fig.colorbar.set_axis_label_text(r'Integrated Intensity (mJy beam$^{-1}$ km s$^{-1}$)')

    :param file_name: string with filename to process
    :return: hdu
    """
    data, hd = fits.getdata(file_name, header=True)
    return fits.PrimaryHDU(data=data*1e3, header=hd)
