import numpy as np
import sys
from astropy import units as u, constants as const
from astropy.table import Table, vstack, join
from astropy.time import Time
from astropy.io import ascii
from astropy.timeseries import TimeSeries, aggregate_downsample
#import plot_figures as pf
from scipy.optimize import curve_fit as cf
from matplotlib import pyplot as plt
import paths

import matplotlib
matplotlib.use('MacOSX')

rose = "#CC6677"
wine = '#882255'
teal = '#44AA99'
green = '#117733'
olive = '#999933'
cyan = '#88CCEE'

def plot_errorbar(x, y_all, y_all_err=None, linelabels="", xhlines=None, xhlinelabels=None, xlabel="default",
                  xscale="linear",yscale="linear",ylabel="default", figsize=(10,8), plottitle=None, axvspans_min=None,
                  xylines_x=None, xylines_y=None, xylinelabels="", colours=None,
                  axvspans_max=None, linestyle="None", linewidth=1, yinvert=False, fig=None, ax=None, cs=2, ms=4):
    """Plot errorbar-plot.

    Arguments:
    If no fig and ax instance is given, create new fig and ax
        - Optionally change figsize
    If fig and ax is provide, use provided fig and ax
    Default is x and y labels is columnnames if available, else "provide label"
    Optionally change xscale (default is linear), yscale (default is linear)
    Optionally change linestyle, default is "None"
    Optional change linewidth, default is 1
    Optionally provide plottitle
    Optionally add hlines
    Optionally add vspan
    Optionally add a lineplot
    Optionally set yinvert to True to invert y-axis

    Return fig and ax
    """
    if fig is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if type(y_all) != list:
        y_all = [y_all]
    if y_all_err is None:
        y_all_err = [None] * len(y_all)
    if type(y_all_err) != list:
        y_all_err = [y_all_err]
    if type(linelabels) != list:
        linelabels = [linelabels]
    if type(x) != list:
        if colours is not None:
            #ax.set_prop_cycle(color=colours)
            for y, y_err, label, c in zip(y_all, y_all_err, linelabels, colours):
                ax.errorbar(x, y, yerr=y_err, linestyle=linestyle, linewidth=linewidth,
                            marker=".", capsize=cs, markersize=ms, label=label, c=c)
        else:
            for y, y_err, label in zip(y_all, y_all_err, linelabels):
                ax.errorbar(x, y, yerr=y_err, linestyle=linestyle, linewidth=linewidth,
                            marker=".", capsize=cs, markersize=ms, label=label)
    else:
        for x_p, y, y_err, label in zip(x, y_all, y_all_err, linelabels):
            ax.errorbar(x_p, y, yerr=y_err, linestyle=linestyle, linewidth=linewidth,
                        marker=".", capsize=cs, markersize=ms, label=label)
    # colours for spans and lines
    colours_line = ["xkcd:dull red","xkcd:sage", "xkcd:orange yellow", "xkcd:coral pink", "xkcd:bluish green", "xkcd:teal blue"]
    if xhlines is not None:
        if type(xhlines) != list:
            xhlines = [xhlines]
        for i, (xhline, xhlabel) in enumerate(zip(xhlines, xhlinelabels)):
            ax.axhline(y=xhline, linestyle="--", label=xhlabel, alpha=0.6, c=colours_line[i])
    if axvspans_min is not None:
        if type(axvspans_min) != list:
            axvspans_min = [axvspans_min]
        if type(axvspans_max) != list:
            axvspans_max = [axvspans_max]
        for i, (axvspan_min, axvspan_max) in enumerate(zip(axvspans_min, axvspans_max)):
            ax.axvspan(axvspan_min, axvspan_max, alpha=0.2, color=colours_line[i])
    if xylines_x is not None:
        if type(xylines_x) != list:
            xylines_x = [xylines_x]
        if type(xylines_y) != list:
            xylines_y = [xylines_y]
        for i, (x_line,y_line,label_line) in enumerate(zip(xylines_x, xylines_y, xylinelabels)):
            ax.plot(x_line, y_line, label=label_line, alpha=0.9, c=colours_line[i], lw=2)
    if xlabel == "default":
        try:
            xlabel = x.name
        except:
            xlabel = "provide xlabel"
    if ylabel == "default":
        try:
            ylabel  = y_all[0].name
        except:
            ylabel = "provide ylabel"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if yinvert==True:
        ax.invert_yaxis()
    plt.legend(loc="best")
    fig.suptitle(plottitle, fontsize=18)
    return fig, ax



def remove_outliers(array, factor=3):
    """Remove outliers in errors with interquantile range.
    """
    q1 = np.nanquantile(array, 0.25)
    q3 = np.nanquantile(array, 0.75)
    iqr = q3 - q1
    #lower_bound = q1 - (iqr * factor)
    upper_bound = q3 + (iqr * factor)
    print("median error: ", np.round(np.nanmedian(array), 2))
    print("upper bound: ", np.round(upper_bound, 2))
    return array < upper_bound


def read_neowise():
    """Read NEOWISE data.
    Add columns with flux and flux error in mJy.
    Compute binned timeseries.
    Return data and binned data.
    """
    data = TimeSeries.read(paths.data / 'rvc_data/neowise.csv', time_column='mjd', time_format="mjd")
    data.sort("time")
    data = data[data["dec"] > -38.9902]   # remove the 2 outliers (tracking error) from the data
    data = data[data["ra"] < 123.847]     # remove the 2 outliers (tracking error) from the data
    data.keep_columns(['time', 'ra', 'dec', 'sigra', 'sigdec', 'sigradec', 'w1mpro', 'w1sigmpro', 'w1snr', 'w1rchi2', 'w2mpro', 'w2sigmpro', 'w2snr', 'w2rchi2'])
    data.rename_columns(["w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro"], ["w1_mag", "w1_mag_err", "w2_mag", "w2_mag_err"])
    print("length of neowise before removing outliers: ", len(data))
    data = data[remove_outliers(data["w1_mag_err"])]
    data = data[remove_outliers(data["w2_mag_err"])]
    print("length of neowise after removing outliers: ", len(data))
    return data


def read_asassn():
    data = TimeSeries.read(paths.data / 'rvc_data/asassn.csv', time_column='HJD', time_format="jd")
    data.sort("time")
    data.rename_column("flux(mJy)", "flux")
    data.keep_columns(['time', 'flux', 'flux_err', 'Filter', 'mag', 'mag_err'])
    data_v = data[data["Filter"] == "V"]
    data_g = data[data["Filter"] == "g"]
    data_v.remove_columns(["Filter"])
    data_g.remove_columns(["Filter"])
    print("length of asassn v before removing outliers: ", len(data_v))
    data_v = data_v[remove_outliers(data_v["mag_err"])]
    print("length of asassn v after removing outliers: ", len(data_v))
    print("length of asassn g before removing outliers: ", len(data_g))
    data_g = data_g[remove_outliers(data_g["mag_err"])]
    print("length of asassn g after removing outliers: ", len(data_g))
    return data_v, data_g


def read_atlas():
    data = TimeSeries.read(paths.data/'rvc_data/atlas_reduced.txt', time_column='MJD', time_format="mjd", format='ascii')
    data.sort("time")
    data = data[np.logical_and(data["uJy"]>0, data["uJy"]<1000000)] # remove NaN values
    data.rename_columns(["m", "dm"], ["mag", "mag_err"])
    data["flux"] = (data["uJy"]*u.uJy).to(u.mJy)        # flux density in mJy
    data["flux_err"] = (data["duJy"]*u.uJy).to(u.mJy)   # flux density error in mJy
    data.keep_columns(['time', 'flux', 'flux_err', 'mag', 'mag_err', 'F'])
    data_c = data[data["F"] == "c"]
    data_o = data[data["F"] == "o"]
    data_c.remove_columns(["F"])
    data_o.remove_columns(["F"])
    print("length of atlas c before removing outliers: ", len(data_c))
    data_c = data_c[remove_outliers(data_c["mag_err"])]
    print("length of atlas c after removing outliers: ", len(data_c))
    print("length of atlas o before removing outliers: ", len(data_o))
    data_o = data_o[remove_outliers(data_o["mag_err"])]
    print("length of atlas o after removing outliers: ", len(data_o))
    return data_c, data_o


def read_aavso():
    """Read ASASSN-21qj data from the aavso survey and normalise to star flux.
    """
    data = TimeSeries.read(paths.data/'rvc_data/aavso.txt', time_column='JD', time_format="jd", format='ascii')
    data.sort("time")
    print(data.colnames)
    data["flux"] = np.power(10, (-data["Magnitude"])*0.4)
    data["flux_err"] = data["Uncertainty"] # rough approximation of normalised flux w Taylor expansion
    data.rename_columns(["Magnitude", "Uncertainty"], ["mag", "mag_err"])
    data.keep_columns(['time', 'flux', 'flux_err', 'mag', 'mag_err', 'Band'])
    data_i = data[data["Band"]=="I"]
    data_v = data[data["Band"]=="V"]
    data_b = data[data["Band"]=="B"]
    data_v.remove_columns(["Band"])
    data_b.remove_columns(["Band"])
    data_i.remove_columns(["Band"])
    print("length of aavso i before removing outliers: ", len(data_i))
    data_i = data_i[remove_outliers(data_i["mag_err"])]
    print("length of aavso i after removing outliers: ", len(data_i))
    print("length of aavso v before removing outliers: ", len(data_v))
    data_v = data_v[remove_outliers(data_v["mag_err"])]
    print("length of aavso v after removing outliers: ", len(data_v))
    print("length of aavso b before removing outliers: ", len(data_b))
    data_b = data_b[remove_outliers(data_b["mag_err"])]
    print("length of aavso b after removing outliers: ", len(data_b))
    return data_i, data_v, data_b


def read_lcogt():
    #data = TimeSeries.read("./data/lcogt/source_extracted/dynamic_masked/dynamic_masked_asassn_table.fits", time_column="time", time_format="mjd")
    data = TimeSeries.read(paths.data/'rvc_data/lcogt.fits', time_column="time", time_format="mjd")
    # remove bad frames
    data = data[data["bad_frame"] < 2] # only flag 0 and 1
    # remove data with nans
    data = data[np.isfinite(data["mag"])]
    # split up the different filters
    data_g = data[data["filter"] == "g"]
    data_i = data[data["filter"] == "i"]
    data_r = data[data["filter"] == "r"]
    print("length of lcogt g before removing outliers: ", len(data_g))
    data_g = data_g[remove_outliers(data_g["mag_err"])]
    print("length of lcogt g after removing outliers: ", len(data_g))
    print("length of lcogt i before removing outliers: ", len(data_i))
    data_i = data_i[remove_outliers(data_i["mag_err"])]
    print("length of lcogt i after removing outliers: ", len(data_i))
    print("length of lcogt r before removing outliers: ", len(data_r))
    data_r = data_r[remove_outliers(data_r["mag_err"])]
    print("length of lcogt r after removing outliers: ", len(data_r))
    return data_g, data_i, data_r


def func(x, b):
    """Linear curve: y=ax+b
    slope = 0
    """
    a = 0
    return a * x + b


def hline_fit(time, flux, flux_error):
    """
    Fit a linear fit with slope of 0 to data.
    Return intercept value.
    """
    p_opt, p_cov = cf(f=func, xdata=time, ydata=flux, sigma=flux_error)
    p_err = np.sqrt(np.diag(p_cov))  # get standard deviation on slope and intercept
    return p_opt[0], p_err[0]


def plot_norms(data_x, data_y, error_y, fit_mag, fit_mag_err, catalog, passband):
    """Plot normalisation of filter.
    """
    fig, ax = plt.subplots(figsize=(10,5))
    fig, ax = plot_errorbar(data_x, data_y, error_y, ax=ax, fig=fig, linelabels="normalised magnitude",
                                xhlines=[fit_mag], xhlinelabels=[f"fit: {fit_mag:.3f} $\pm$ {fit_mag_err:.3f}"],
                               plottitle=f"Normalised {catalog} {passband}-magnitude", yinvert=True,
                               colours=["xkcd:dark blue grey"], xlabel="time [MJD]", ylabel="magnitude")
    #pf.save_figure(fig, subdir="lightcurves/master_lightcurve/normalisations/", filename=f"{catalog}_{passband}")
    fig.savefig("_asdf3.pdf")
    plt.close()


def neowise_norms(data):
    """Fit line to w1 and w2 magnitudes.
    """
    max_fit = 58100
    fit_region = data[data.time.mjd<max_fit]
    w1_ic, w1_ic_err = hline_fit(fit_region.time.mjd, fit_region["w1_mag"],fit_region["w1_mag_err"])
    w2_ic, w2_ic_err = hline_fit(fit_region.time.mjd, fit_region["w2_mag"],fit_region["w2_mag_err"])
    # normalised fluxes
    data["w1_mag_norm"] = data["w1_mag"] - w1_ic
    data["w2_mag_norm"] = data["w2_mag"] - w2_ic
    # plot normalisations
    plot_norms(data.time.mjd, data["w1_mag"], data["w1_mag_err"], w1_ic, w1_ic_err, "neowise", "w1")
    plot_norms(data.time.mjd, data["w2_mag"], data["w2_mag_err"], w2_ic, w2_ic_err, "neowise", "w2")
    print("neowise w1 normalisation: ", w1_ic)
    print("neowise w2 normalisation: ", w2_ic)
    return data


def asassn_norms(data_v, data_g):
    """Fit line to asassn g and V flux.
    """
    max_fit = 58700
    fit_region = data_v[data_v.time.mjd<max_fit]
    v_ic, v_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    fit_region = data_g[data_g.time.mjd<max_fit]
    g_ic, g_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    # compute normalised fluxes
    data_v["mag_norm"] = data_v["mag"] - v_ic
    data_g["mag_norm"] = data_g["mag"] - g_ic
    # plot normalisations
    plot_norms(data_v.time.mjd, data_v["mag"], data_v["mag_err"], v_ic, v_ic_err, "asassn", "v")
    plot_norms(data_g.time.mjd, data_g["mag"], data_g["mag_err"], g_ic, g_ic_err, "asassn", "g")
    print("asassn v normalisation: ", v_ic)
    print("asassn g normalisation: ", g_ic)
    global asassn_v_norm_constant, asassn_g_norm_constant
    asassn_v_norm_constant = v_ic
    asassn_g_norm_constant = g_ic
    return data_v, data_g


def atlas_norms(data_c, data_o):
    """Fit line to atlas c and o magnitudes.
    """
    max_fit = 58700
    fit_region = data_c[np.logical_and(data_c.time.mjd<max_fit, data_c["mag"].value > 5)]
    c_ic, c_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    fit_region = data_o[np.logical_and(data_o.time.mjd<max_fit, data_o["mag"].value > 5)]
    o_ic, o_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    # compute normalised mags
    data_c["mag_norm"] = data_c["mag"] - c_ic
    data_o["mag_norm"] = data_o["mag"] - o_ic
    # plot normalisations
    plot_norms(data_c.time.mjd, data_c["mag"], data_c["mag_err"], c_ic, c_ic_err, "atlas", "c")
    plot_norms(data_o.time.mjd, data_o["mag"], data_o["mag_err"], o_ic, o_ic_err, "atlas", "o")
    # compute difference end mag and beginning mag
    min_fit = 59800
    min_c = np.min(data_c[data_c.time.mjd>min_fit]["mag"])
    min_o = np.min(data_o[data_o.time.mjd>min_fit]["mag"])
    print(f"min c: {min_c:.3f}, min o: {min_o:.3f}")
    print("c diff", min_c - c_ic)
    print("o diff", min_o - o_ic)
    print("atlas c normalisation: ", c_ic)
    print("atlas o normalisation: ", o_ic)
    global atlas_c_norm_constant, atlas_o_norm_constant
    atlas_c_norm_constant = c_ic
    atlas_o_norm_constant = o_ic
    return data_c, data_o


def aavso_norms(data_i, data_v, data_b):
    """Fit line to aavso g, i, u, r magnitudes.
    """
    min_fit = 59800
    fit_region = data_i[data_i.time.mjd>min_fit]
    i_ic, i_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    fit_region = data_v[data_v.time.mjd>min_fit]
    v_ic, v_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    fit_region = data_b[data_b.time.mjd>min_fit]
    b_ic, b_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    data_i["mag_norm"] = data_i["mag"] - i_ic
    data_v["mag_norm"] = data_v["mag"] - v_ic
    data_b["mag_norm"] = data_b["mag"] - b_ic
    # plot normalisations
    plot_norms(data_i.time.mjd, data_i["mag"], data_i["mag_err"], i_ic, i_ic_err, "aavso", "i")
    plot_norms(data_v.time.mjd, data_v["mag"], data_v["mag_err"], v_ic, v_ic_err, "aavso", "v")
    plot_norms(data_b.time.mjd, data_b["mag"], data_b["mag_err"], b_ic, b_ic_err, "aavso", "b")
    # add offset (because flux is not fully back to normal after the eclipse)
    data_i["mag_norm"] += 0.19
    data_v["mag_norm"] += 0.18
    data_b["mag_norm"] += 0.14
    print("aavso i normalisation: ", i_ic-0.19)
    print("aavso v normalisation: ", v_ic-0.18)
    print("aavso b normalisation: ", b_ic-0.14)
    global aavso_i_norm_constant, aavso_v_norm_constant, aavso_b_norm_constant
    aavso_i_norm_constant = i_ic-0.19
    aavso_v_norm_constant = v_ic-0.18
    aavso_b_norm_constant = b_ic-0.14
    return data_i, data_v, data_b


def lcogt_norms(data_g, data_i, data_r):
    """Fit line to lcogt g, i and r magnitudes.
    """
    min_fit = 59800
    fit_region = data_g[data_g.time.mjd>min_fit]
    g_ic, g_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    fit_region = data_i[data_i.time.mjd>min_fit]
    i_ic, i_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    # no data in fit region for r
    #fit_region = data_r[data_r.time.mjd>min_fit]
    #r_ic, r_ic_err = hline_fit(fit_region.time.mjd, fit_region["mag"],fit_region["mag_err"])
    data_g["mag_norm"] = data_g["mag"] - g_ic
    data_i["mag_norm"] = data_i["mag"] - i_ic
    #data_r["mag_norm"] = data_r["mag"] - r_ic
    # plot normalisations
    plot_norms(data_g.time.mjd, data_g["mag"], data_g["mag_err"], g_ic, g_ic_err, "lcogt", "g")
    #plot_norms(data_r.time.mjd, data_r["mag"], data_r["mag_err"], r_ic, r_ic_err, "lcogt", "r")
    plot_norms(data_i.time.mjd, data_i["mag"], data_i["mag_err"], i_ic, i_ic_err, "lcogt", "i")
    # compute temp r
    r_ic = np.min(data_r["mag"])
    data_r["mag_norm"] = data_r["mag"] - r_ic
    plot_norms(data_r.time.mjd, data_r["mag"], data_r["mag_err"], r_ic, 0, "lcogt", "r")
    # add offset (because flux is not fully back to normal after the eclipse)
    data_g["mag_norm"] += 0.22  # note that these calibrations indicate mag is currently 0.2 higher than normal
    data_r["mag_norm"] += 0.15
    data_i["mag_norm"] += 0.19
    print("lcogt g normalisation: ", g_ic-0.22)
    print("lcogt r normalisation: ", r_ic-0.15)
    print("lcogt i normalisation: ", i_ic-0.19)
    global lcogt_g_norm_constant, lcogt_r_norm_constant, lcogt_i_norm_constant
    lcogt_g_norm_constant = g_ic-0.22
    lcogt_r_norm_constant = r_ic-0.15
    lcogt_i_norm_constant = i_ic-0.19
    return data_g, data_i, data_r


def bin_neowise(data):
    """Return binned timeseries.
    """
    binwidth = 50
    bincenters = np.unique(binwidth*np.round(data.time.mjd/binwidth, decimals=0))
    start_time = Time(bincenters-(binwidth/2.), format="mjd")
    end_time = Time(bincenters+(binwidth/2.), format="mjd")
    data_binned = aggregate_downsample(data, time_bin_start=start_time, time_bin_end=end_time, aggregate_func=np.nanmedian)
    data_std_binned = aggregate_downsample(data, time_bin_start=start_time, time_bin_end=end_time, aggregate_func=np.nanstd)
    # replace median of standard deviation with standard deviation of flux in bin
    data_binned["w1_mag_err"] = np.fmax(data_binned["w1_mag_err"], data_std_binned["w1_mag_norm"])
    data_binned["w2_mag_err"] = np.fmax(data_binned["w2_mag_err"], data_std_binned["w2_mag_norm"])
    plot_norms(data_binned.time_bin_center.mjd, data_binned["w1_mag_norm"], data_binned["w1_mag_err"], 0, 0, "neowise_binned", "w1")
    plot_norms(data_binned.time_bin_center.mjd, data_binned["w2_mag_norm"], data_binned["w2_mag_err"], 0, 0, "neowise_binned", "w2")
    return data_binned


def bin_timeseries(data, binwidth=1):
    """Downsample to bins with width binwidth. Default is 1 day.
    Return binned timeseries.
    """
    data_binned = aggregate_downsample(data, time_bin_size=binwidth*u.day, aggregate_func=np.nanmedian)
    data_std_binned = aggregate_downsample(data, time_bin_size=binwidth*u.day, aggregate_func=np.nanstd)
    # replace median of standard deviation with standard deviation of flux in bin
    data_binned["mag_err"] = np.fmax(data_binned["mag_err"], data_std_binned["mag_norm"])   # whichever is bigger
    # remove bins with no content
    data_binned = data_binned.filled(fill_value=np.nan)
    data_binned = data_binned[np.isfinite(data_binned["mag_norm"])]
    return data_binned


def masterplot(neowise, aavso_b, aavso_v, aavso_i, asassn_v, asassn_g, atlas_c, atlas_o, lcogt_g, lcogt_i, lcogt_r):
    """Big plot"""
    cs = 1
    ms = 2
    # markers and colours
    nw_marker = "s"
    aavso_marker = "v"
    asassn_marker = "o"
    atlas_marker = "D"
    lcogt_marker = "^"
    g1_colour = "xkcd:wine red"
    g2_colour = "xkcd:reddish gray"
    g3_colour = "xkcd:faded red"
    g4_colour = teal
    g5_colour = cyan

    fig, ax = plt.subplots(1, 1, figsize=(80, 10))
    # neowise commented out for more vertical details
    # group 1 (reddest)
    # offset_group1 = -4
    # neowise["w2_mag_norm"] += offset_group1
    # ax.errorbar(neowise.time_bin_center.mjd, neowise["w2_mag_norm"], yerr=neowise["w2_mag_err"], ls="none",
    #             marker=nw_marker, color=g1_colour, label="NEOWISE w2-band",
    #             capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    # ax.axhline(offset_group1, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # # group 2
    # offset_group2 = -2
    # neowise["w1_mag_norm"] += offset_group2
    # ax.errorbar(neowise.time_bin_center.mjd, neowise["w1_mag_norm"], yerr=neowise["w1_mag_err"], ls="none",
    #             marker=nw_marker, color=g2_colour, label="NEOWISE w1-band",
    #             capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    # ax.axhline(offset_group2, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 3
    offset_group3 = 0
    aavso_i["mag_norm"] += offset_group3
    lcogt_i["mag_norm"] += offset_group3
    atlas_o["mag_norm"] += offset_group3
    ax.errorbar(aavso_i.time_bin_center.mjd, aavso_i["mag_norm"], yerr=aavso_i["mag_err"], ls="none",
                marker=aavso_marker, color=g3_colour, label="AAVSO I-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(lcogt_i.time.mjd, lcogt_i["mag_norm"], yerr=lcogt_i["mag_err"], ls="none",
                marker=lcogt_marker, color=g3_colour, label="LCOGT i-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(atlas_o.time_bin_center.mjd, atlas_o["mag_norm"], yerr=atlas_o["mag_err"], ls="none",
                marker=atlas_marker, color=g3_colour, label="ATLAS o-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.axhline(offset_group3, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 4
    offset_group4 = 2
    aavso_v["mag_norm"] += offset_group4
    asassn_v["mag_norm"] += offset_group4
    lcogt_r["mag_norm"] += offset_group4
    atlas_c["mag_norm"] += offset_group4
    ax.errorbar(aavso_v.time_bin_center.mjd, aavso_v["mag_norm"], yerr=aavso_v["mag_err"], ls="none",
                marker=aavso_marker, color=g4_colour, label="AAVSO V-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(asassn_v.time.mjd, asassn_v["mag_norm"], yerr=asassn_v["mag_err"], ls="none",
                marker=asassn_marker, color=g4_colour, label="ASASSN V-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(lcogt_r.time.mjd, lcogt_r["mag_norm"], yerr=lcogt_r["mag_err"], ls="none",
                marker=lcogt_marker, color=g4_colour, label="LCOGT r-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(atlas_c.time_bin_center.mjd, atlas_c["mag_norm"], yerr=atlas_c["mag_err"], ls="none",
                marker=atlas_marker, color=g4_colour, label="ATLAS c-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.axhline(offset_group4, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 5
    offset_group5 = 4
    aavso_b["mag_norm"] += offset_group5
    asassn_g["mag_norm"] += offset_group5
    lcogt_g["mag_norm"] += offset_group5
    ax.errorbar(aavso_b.time_bin_center.mjd, aavso_b["mag_norm"], yerr=aavso_b["mag_err"], ls="none",
                marker=aavso_marker, color=g5_colour, label="AAVSO B-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(asassn_g.time.mjd, asassn_g["mag_norm"], yerr=asassn_g["mag_err"], ls="none",
                marker=asassn_marker, color=g5_colour, label="ASASSN g-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(lcogt_g.time.mjd, lcogt_g["mag_norm"], yerr=lcogt_g["mag_err"], ls="none",
                marker=lcogt_marker, color=g5_colour, label="LCOGT g-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.axhline(offset_group5, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # brightening event
    ax.axvspan(58055, 58245, alpha=0.15, color="xkcd:light peach", zorder=1)
    ax.annotate("Brightening event", xy=(58150, -0.3), xytext=(58150, -0.3), fontsize=16, c="xkcd:dark peach", zorder=3,
                ha="center", va="center")
    # plot properties
    ax.invert_yaxis()
    #ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticks(np.arange(57900, 60000, 100))
    #ax.set_xlim(left=57820, right=59950)
    ax.set_xlim(left=58000, right=59950)
    ax.set_ylim(top=-0.6, bottom=7.2)
    ax.set_xlabel("epoch [MJD]")
    ax.set_ylabel("normalised magnitude")
    ax.minorticks_on()
    ax.grid(color="xkcd:cool grey", alpha=0.5, which='major', zorder=0)
    ax.grid(color="xkcd:cool grey", alpha=0.1, which='minor', zorder=0)
    ax.legend(loc="lower right", title="Bandpass filter", title_fontsize=16, framealpha=0.95, fontsize=12, markerscale=3)
    # legend on the side
    #leg = ax.legend(markerscale=3, title="Catalog", title_fontsize=12, bbox_to_anchor=(1.0, 0, 0, 1.016),loc='upper left',ncol=1,
    #                    edgecolor="black", borderaxespad=1.0)
    #plt.tight_layout()
    #fontsize = fig.canvas.get_renderer().points_to_pixels(leg._fontsize)
    #pad = 1.4 * (leg.borderaxespad + leg.borderpad) * fontsize
    #leg._legend_box.set_height(leg.get_bbox_to_anchor().height-pad)


    # pf.save_figure(fig, subdir="/lightcurves/master_lightcurve/", filename="master_curve", ext=".png", dpi=300)
    fig.savefig('_asdf.pdf')
    plt.close()

def masterplot_paper(neowise, aavso_b, aavso_v, aavso_i, asassn_v, asassn_g, atlas_c, atlas_o, lcogt_g, lcogt_i, lcogt_r):
    """Big plot in A4 size for paper.
    """
    cs = 1
    ms = 2
    # markers and colours
    nw_marker = "s"
    aavso_marker = "v"
    asassn_marker = "o"
    atlas_marker = "D"
    lcogt_marker = "^"
    g1_colour = "xkcd:wine red"
    g2_colour = "xkcd:reddish gray"
    g3_colour = "xkcd:faded red"
    g4_colour = teal
    g5_colour = cyan

    fig, (ax, ax3) = plt.subplots(2, 1, figsize=(13, 7),sharex=True)
    # neowise commented out for more vertical details
    # group 1 (reddest)
    # offset_group1 = -4
    # neowise["w2_mag_norm"] += offset_group1
    # ax.errorbar(neowise.time_bin_center.mjd, neowise["w2_mag_norm"], yerr=neowise["w2_mag_err"], ls="none",
    #             marker=nw_marker, color=g1_colour, label="NEOWISE w2-band",
    #             capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    # ax.axhline(offset_group1, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # # group 2
    # offset_group2 = -2
    # neowise["w1_mag_norm"] += offset_group2
    # ax.errorbar(neowise.time_bin_center.mjd, neowise["w1_mag_norm"], yerr=neowise["w1_mag_err"], ls="none",
    #             marker=nw_marker, color=g2_colour, label="NEOWISE w1-band",
    #             capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    # ax.axhline(offset_group2, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 3
    offset_group3 = 0
    aavso_i["mag_norm"] += offset_group3
    lcogt_i["mag_norm"] += offset_group3
    atlas_o["mag_norm"] += offset_group3
    ax.errorbar(aavso_i.time_bin_center.mjd, aavso_i["mag_norm"], yerr=aavso_i["mag_err"], ls="none",
                marker=aavso_marker, color=g3_colour, label="AAVSO I-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(lcogt_i.time.mjd, lcogt_i["mag_norm"], yerr=lcogt_i["mag_err"], ls="none",
                marker=lcogt_marker, color=g3_colour, label="LCOGT i-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(atlas_o.time_bin_center.mjd, atlas_o["mag_norm"], yerr=atlas_o["mag_err"], ls="none",
                marker=atlas_marker, color=g3_colour, label="ATLAS o-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.axhline(offset_group3, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 4
    offset_group4 = 2   # already done in previous function
    #aavso_v["mag_norm"] += offset_group4
    #asassn_v["mag_norm"] += offset_group4
    #lcogt_r["mag_norm"] += offset_group4
    #atlas_c["mag_norm"] += offset_group4
    ax.errorbar(aavso_v.time_bin_center.mjd, aavso_v["mag_norm"], yerr=aavso_v["mag_err"], ls="none",
                marker=aavso_marker, color=g4_colour, label="AAVSO V-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(asassn_v.time.mjd, asassn_v["mag_norm"], yerr=asassn_v["mag_err"], ls="none",
                marker=asassn_marker, color=g4_colour, label="ASASSN V-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(lcogt_r.time.mjd, lcogt_r["mag_norm"], yerr=lcogt_r["mag_err"], ls="none",
                marker=lcogt_marker, color=g4_colour, label="LCOGT r-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.errorbar(atlas_c.time_bin_center.mjd, atlas_c["mag_norm"], yerr=atlas_c["mag_err"], ls="none",
                marker=atlas_marker, color=g4_colour, label="ATLAS c-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.axhline(offset_group4, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # group 5
    offset_group5 = 4   # already done in previous function
    #aavso_b["mag_norm"] += offset_group5
    #asassn_g["mag_norm"] += offset_group5
    #lcogt_g["mag_norm"] += offset_group5
    ax.errorbar(aavso_b.time_bin_center.mjd, aavso_b["mag_norm"], yerr=aavso_b["mag_err"], ls="none",
                marker=aavso_marker, color=g5_colour, label="AAVSO B-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5, rasterized=True)
    ax.errorbar(asassn_g.time.mjd, asassn_g["mag_norm"], yerr=asassn_g["mag_err"], ls="none",
                marker=asassn_marker, color=g5_colour, label="ASASSN g-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5, rasterized=True)
    ax.errorbar(lcogt_g.time.mjd, lcogt_g["mag_norm"], yerr=lcogt_g["mag_err"], ls="none",
                marker=lcogt_marker, color=g5_colour, label="LCOGT g-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5, rasterized=True)
    ax.axhline(offset_group5, c="xkcd:dusk", alpha=0.7, ls="--", zorder=2)
    # brightening event
    ax.axvspan(58055, 58245, alpha=0.15, color="xkcd:light peach", zorder=1)
    ax.annotate("Brightening event", xy=(58150, -0.8), xytext=(58150, -0.8), fontsize=16, c="xkcd:dark peach", zorder=3,
                ha="center", va="center")
    # plot properties
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.invert_yaxis()
    #ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticks(np.arange(57900, 60000, 100))
    #ax.set_xlim(left=57820, right=59950)
#    ax.set_xlim(left=59100, right=59950)
    ax.set_ylabel("normalised magnitude", fontsize=16)
    ax.minorticks_on()
    ax.grid(color="xkcd:cool grey", alpha=0.5, which='major', zorder=0)
    ax.grid(color="xkcd:cool grey", alpha=0.1, which='minor', zorder=0)
    ax3.grid(color="xkcd:cool grey", alpha=0.5, which='major', zorder=0)
    ax3.grid(color="xkcd:cool grey", alpha=0.1, which='minor', zorder=0)
    ax.legend(loc="lower left", title="Bandpass filter", title_fontsize=11, framealpha=0.95, fontsize=9, markerscale=2)
    # legend on the side

    #ax2.errorbar(fit_x,np.abs(fit_m),yerr=fit_merr,fmt='w.',mec='black',alpha=0.3)
    #ax2.errorbar(fit_x[mh],np.abs(fit_m[mh]),yerr=fit_merr[mh],fmt='k.')

    ax3.errorbar(fit_x,vtrans.value,yerr=vtrans_err.value,fmt='k.',mec='black')
 #   ax3.errorbar(fit_x[mh],vtrans.value[mh],yerr=vtrans_err.value[mh],fmt='kv')


    #ax.set_ylabel('Normalised flux')
    ax3.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
 #   ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
 #   ax.set_xlabel("Epoch [MJD]", fontsize=16)
    ax3.set_xticks(np.arange(57900, 60000, 100))
    ax.set_xticks(np.arange(57900, 60000, 100))
    ax3.minorticks_on()

    ax3.set_xlabel('Epoch [MJD]',fontsize=16)
    #ax2.set_ylabel('Gradient of flux [$d^{-1}$]')
    ax3.set_ylabel('$v$ [$km\ s^{-1}$]', fontsize=16)
    ax3.set_xlim(59660,59760) 


    plt.tight_layout()
    fig.savefig(paths.figures / 'master_lightcurve_nature_with_zoom_3_panel.pdf')
    plt.close()



def plot_bump(asassn_v, atlas_o):
    """Plot zoom of small bump in brightening event.
    """
    # markers and colours
    cs = 1
    ms = 2
    nw_marker = "s"
    aavso_marker = "v"
    asassn_marker = "o"
    atlas_marker = "D"
    lcogt_marker = "^"
    g1_colour = "xkcd:wine red"
    g2_colour = "xkcd:reddish gray"
    g3_colour = "xkcd:faded red"
    g4_colour = teal
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(asassn_v.time_bin_center.mjd, asassn_v["mag"], yerr=asassn_v["mag_err"], ls="none",
                marker=asassn_marker, color=g4_colour, label="ASASSN V-band",
                capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    #ax.errorbar(atlas_o.time.mjd, atlas_o["mag_norm"], yerr=atlas_o["mag_err"], ls="none",
    #            marker=atlas_marker, color=g3_colour, label="ATLAS o-band",
    #            capsize=cs, markersize=ms, zorder=3, linewidth=0.5, capthick=0.5)
    ax.set_xlim(left=58140, right=58255)

    ax.grid(color="xkcd:cool grey", alpha=0.5, zorder=0)
    pf.save_figure(fig, subdir="/lightcurves/master_lightcurve/", filename="brightening_bump_v_bin", ext=".png", dpi=300)
    plt.close()


def combine_i_catalogs(aavso_i, lcogt_i, atlas_o):
    """Combine all 'group 3' catalogs (~i band) into 1 catalog.
    Write to csv and return catalog.
    """
    aavso_i["time"] = aavso_i.time_bin_center
    aavso_i = aavso_i["time", "mag_norm", "mag_err"]
    aavso_i["cat"] = "aavso-i"
    aavso_i["norm"] = aavso_i_norm_constant
    lcogt_i = lcogt_i["time", "mag_norm", "mag_err"]
    lcogt_i["cat"] = "lcogt-i"
    lcogt_i["norm"] = lcogt_i_norm_constant
    atlas_o["time"] = atlas_o.time_bin_center
    atlas_o = atlas_o["time", "mag_norm", "mag_err"]
    atlas_o["cat"] = "atlas-o"
    atlas_o["norm"] = atlas_o_norm_constant
    i_cat = vstack([aavso_i, lcogt_i, atlas_o])
    ascii.write(i_cat, paths.data / 'all_i.csv', format='csv', overwrite=True)
    return i_cat


def combine_v_catalogs(asassn_v, aavso_v, atlas_c, lcogt_r):
    """Combine all 'group 4' catalogs (~v band) into 1 catalog.
    Write to csv and return catalog.
    """
    asassn_v = asassn_v["time", "mag_norm", "mag_err"]
    asassn_v["cat"] = "asassn-v"
    asassn_v["norm"] = asassn_v_norm_constant
    aavso_v["time"] = aavso_v.time_bin_center
    aavso_v = aavso_v["time", "mag_norm", "mag_err"]
    aavso_v["cat"] = "aavso-v"
    aavso_v["norm"] = aavso_v_norm_constant
    atlas_c["time"] = atlas_c.time_bin_center
    atlas_o = atlas_c["time", "mag_norm", "mag_err"]
    atlas_o["cat"] = "atlas-o"
    atlas_o["norm"] = atlas_o_norm_constant
    lcogt_r = lcogt_r["time", "mag_norm", "mag_err"]
    lcogt_r["cat"] = "lcogt-r"
    lcogt_r["norm"] = lcogt_r_norm_constant
    v_cat = vstack([asassn_v, aavso_v, atlas_o, lcogt_r])
    ascii.write(v_cat, paths.data / 'all_v.csv', format="csv", overwrite=True)
    return v_cat


def combine_g_catalogs(aavso_b, asassn_g, lcogt_g):
    """Combine all 'group 5' catalogs (~g band) into 1 catalog.
    Write to csv and return catalog.
    """
    aavso_b["time"] = aavso_b.time_bin_center
    aavso_b = aavso_b["time", "mag_norm", "mag_err"]
    aavso_b["cat"] = "aavso-b"
    aavso_b["norm"] = aavso_b_norm_constant
    asassn_g = asassn_g["time", "mag_norm", "mag_err"]
    asassn_g["cat"] = "asassn-g"
    asassn_g["norm"] = asassn_g_norm_constant
    lcogt_g = lcogt_g["time", "mag_norm", "mag_err"]
    lcogt_g["cat"] = "lcogt-g"
    lcogt_g["norm"] = lcogt_g_norm_constant
    g_cat = vstack([aavso_b, asassn_g, lcogt_g])
    ascii.write(g_cat, paths.data / 'all_g.csv', format="csv", overwrite=True)
    return g_cat


def read_combined():
    i = TimeSeries.read(paths.data/'all_i.csv', time_column='time', time_format="jd")
    v = TimeSeries.read(paths.data/'all_v.csv', time_column='time', time_format="jd")
    g = TimeSeries.read(paths.data/'all_g.csv', time_column='time', time_format="jd")
    i.sort("time")
    v.sort("time")
    g.sort("time")
    return i, v, g


def bin_combined(data, normto, binwidth=1):
    """Bin combined data with bins of binwidth days.
    """
    # Use AAVSO normalisations (for colour purpose)
    magnorm = np.unique(data["norm"][data["cat"] == normto])
    data["mag"] = data["mag_norm"] + magnorm

    # bin data
    data.remove_columns(["cat", "norm"])
    data = data[data.time.mjd > 58000]
    bin_start = Time(np.arange(58000, 59950, binwidth), format="mjd")
    bin_end = bin_start + binwidth * u.day
    bin_center = bin_start + 0.5 * binwidth * u.day
    zerocol = np.zeros(len(bin_start))
    binned = Table(names=["time", "mag", "mag_norm", "mag_err"],
                   data=[bin_center.mjd, zerocol, zerocol, zerocol],
                   dtype=[float, float, float, float])
    for j in range(len(bin_start)):
        in_bin = data[(data.time.mjd >= bin_start[j].mjd) & (data.time.mjd < bin_end[j].mjd)]
        try:
            w = 1 / in_bin["mag_err"] ** 2                    # weights
            mean_mag = np.average(in_bin["mag"], weights=w)
            mean_mag_norm = np.average(in_bin["mag_norm"], weights=w)
            error = 1 / np.sqrt(np.sum(w))
            binned["mag"][j] = mean_mag
            binned["mag_norm"][j] = mean_mag_norm
            binned["mag_err"][j] = error
        except ZeroDivisionError: # in case there are no measurements in the interval
            binned["mag"][j] = np.nan
            binned["mag_err"][j] = np.nan
            binned["mag_norm"][j] = np.nan

    # save binned
    ascii.write(binned, paths.data/'all_{}_binned.csv'.format(normto[-1]), format="csv", overwrite=True)
    ## remove bins with no content
    #binned = binned.filled(fill_value=np.nan)
    #binned = binned[np.isfinite(binned["mag_norm"])]
    #print(binned)
    return binned


def binned_colours(i, v, g):
    i_tab = Table(i)
    v_tab = Table(v)
    g_tab = Table(g)
    binned = join(i_tab, v_tab, keys=["time"], table_names=["i", "v"])
    g_tab.rename_columns(["mag", "mag_err", "mag_norm"], ["mag_g", "mag_err_g", "mag_norm_g"])
    binned = join(binned, g_tab, keys=["time"])
    binned = binned.filled(fill_value=np.nan)
    binned["v-i"] = binned["mag_v"] - binned["mag_i"]
    binned["b-i"] = binned["mag_g"] - binned["mag_i"]
    binned["b-v"] = binned["mag_g"] - binned["mag_v"]
    binned["v-i_err"] = np.sqrt(binned["mag_err_v"]**2 + binned["mag_err_i"]**2)
    binned["b-i_err"] = np.sqrt(binned["mag_err_g"]**2 + binned["mag_err_i"]**2)
    binned["b-v_err"] = np.sqrt(binned["mag_err_g"]**2 + binned["mag_err_v"]**2)
    binned = binned[np.isfinite(binned["v-i"])]
    ascii.write(binned, paths.data/'all_binned.csv', format="csv", overwrite=True)
    return binned


def main():
    # Read neowise, asassn, atlas, aavso and lcogt data
    neowise = read_neowise()
    asassn_v, asassn_g = read_asassn()
    atlas_c, atlas_o = read_atlas()
    aavso_i, aavso_v, aavso_b = read_aavso()
    lcogt_g, lcogt_i, lcogt_r = read_lcogt()

    # Get normalised magnitudes
    neowise = neowise_norms(neowise)
    asassn_v, asassn_g = asassn_norms(asassn_v, asassn_g)
    atlas_c, atlas_o = atlas_norms(atlas_c, atlas_o)
    aavso_i, aavso_v, aavso_b = aavso_norms(aavso_i, aavso_v, aavso_b)
    lcogt_g, lcogt_i, lcogt_r = lcogt_norms(lcogt_g, lcogt_i, lcogt_r)

    # Get binned timeseries
    neowise_binned = bin_neowise(neowise)
    atlas_c_binned = bin_timeseries(atlas_c)
    atlas_o_binned = bin_timeseries(atlas_o)
    atlas_o_binned = atlas_o_binned[remove_outliers(atlas_o_binned["mag_err"], factor=3)]
    atlas_c_binned = atlas_c_binned[remove_outliers(atlas_c_binned["mag_err"], factor=3)]
    plot_norms(atlas_c_binned.time_bin_center.mjd, atlas_c_binned["mag_norm"], atlas_c_binned["mag_err"], 0, 0, "atlas_binned", "c")
    plot_norms(atlas_o_binned.time_bin_center.mjd, atlas_o_binned["mag_norm"], atlas_o_binned["mag_err"], 0, 0, "atlas_binned", "o")
    aavso_i_binned = bin_timeseries(aavso_i)
    aavso_v_binned = bin_timeseries(aavso_v)
    aavso_b_binned = bin_timeseries(aavso_b)
    aavso_i_binned = aavso_i_binned[remove_outliers(aavso_i_binned["mag_err"], factor=3)]
    aavso_v_binned = aavso_v_binned[remove_outliers(aavso_v_binned["mag_err"], factor=3)]
    aavso_b_binned = aavso_b_binned[remove_outliers(aavso_b_binned["mag_err"], factor=3)]
    plot_norms(aavso_i_binned.time_bin_center.mjd, aavso_i_binned["mag_norm"], aavso_i_binned["mag_err"], 0, 0, "aavso_binned", "i")
    plot_norms(aavso_v_binned.time_bin_center.mjd, aavso_v_binned["mag_norm"], aavso_v_binned["mag_err"], 0, 0, "aavso_binned", "v")
    plot_norms(aavso_b_binned.time_bin_center.mjd, aavso_b_binned["mag_norm"], aavso_b_binned["mag_err"], 0, 0, "aavso_binned", "b")
    #asassn_v_binned = bin_timeseries(asassn_v)
    #asassn_g_binned = bin_timeseries(asassn_g)

    # Combine into mastercatalogs
    i_cat = combine_i_catalogs(aavso_i_binned, lcogt_i, atlas_o_binned)
    v_cat = combine_v_catalogs(asassn_v, aavso_v_binned, atlas_c_binned, lcogt_r)
    g_cat = combine_g_catalogs(aavso_b_binned, asassn_g, lcogt_g)

    # Plot all lightcurves
    masterplot(neowise_binned, aavso_b_binned, aavso_v_binned, aavso_i_binned, asassn_v, asassn_g, atlas_c_binned, atlas_o_binned, lcogt_g, lcogt_i, lcogt_r)
    masterplot_paper(neowise_binned, aavso_b_binned, aavso_v_binned, aavso_i_binned, asassn_v, asassn_g, atlas_c_binned, atlas_o_binned, lcogt_g, lcogt_i, lcogt_r)
    #asassn_v_binned_10d = bin_timeseries(asassn_v, binwidth=10)
    #plot_bump(asassn_v_binned_10d, atlas_o)

    # Combined catalogues
    i_cat, v_cat, g_cat = read_combined()
    i_binned = bin_combined(i_cat, "aavso-i")
    v_binned = bin_combined(v_cat, "aavso-v")
    g_binned = bin_combined(g_cat, "aavso-b")
    binned_all = binned_colours(i_binned, v_binned, g_binned)


import numpy as np
from pathlib import Path
from astropy.table import Table, vstack, unique
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import datetime

from asassn21qj import *

import paths
import matplotlib
matplotlib.use('MacOSX')
print('running fit_2d_turning_points...')

tas = ascii.read(paths.data / 'obs_ASASSN.ecsv')
tav = ascii.read(paths.data / 'obs_AAVSO.ecsv')

tas_by_filter = tas.group_by('Filter')
print('ASASSN all observed photometric bands:')
print(tas_by_filter.groups.keys)

tav_by_filter = tav.group_by('Filter')
print('AAVSO all observed photometric bands:')
print(tav_by_filter.groups.keys)

# order should be....
tavB = tav[tav['Filter']=='B']
tasg = tas[tas['Filter']=='g']
tavV = tav[tav['Filter']=='V']
tavI = tav[tav['Filter']=='I']
offset = 0.6

# read in turning points
inf = 'turning_points_g2.txt'

t = Table.read(paths.scripts / inf,format='ascii.no_header')
xc=t['col1']
yc=t['col2']

fig2, axes = plt.subplots(3,1,figsize=(10,6),sharex=True)
# #fig, axes = plt.subplots(3,1,figsize=(10,6))

(fax, fax2,fax3) = np.ndarray.flatten(axes)

fax.errorbar(tasg['MJD'],tasg['fnorm'], yerr=tasg['fnormerr'],fmt='.',color='black',markersize=4 )

fax.vlines(xc,-1,2,color='black',alpha=0.1,linewidth=1)
fax2.vlines(xc,-1,20,color='black',alpha=0.1,linewidth=1)

fax.axhline(0,color='black',alpha=0.1,linestyle='dotted')
fax2.axhline(0,color='black',alpha=0.1,linestyle='dotted')
fax3.axhline(0,color='black',alpha=0.1,linestyle='dotted')
fax.axhline(1,color='black',alpha=0.1,linestyle='dotted')

#fax.plot(xc,yc,'r')

fax.set_ylim(-0.1,1.1)

def linfit_hogg(x,y,yerr):
    'linfit_hogg - from https://emcee.readthedocs.io/en/stable/tutorials/line/'
    A = np.vander(x, 2)
    C = np.diag(yerr * yerr)
    ATA = np.dot(A.T, A / (yerr**2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
    print("Least-squares estimates:")
    m = w[0]
    merr = np.sqrt(cov[0,0])
    b = w[1]
    berr = np.sqrt(cov[1,1])
    print("m = {0:.5f} ± {1:.5f}".format(m,merr))
    print("b = {0:.3f} ± {1:.3f}".format(b,berr))
    return (m,merr,b,berr)

fit_x = []
fit_m = []
fit_merr = []
fit_width = []

x_lower = []
x_upper = []
y_lower = []
y_upper = []

max_tau = [] # as defined in Kennedy 2017 eq. 4.5
for i in np.arange(xc.size-1):
    x1 = xc[i]
    x2 = xc[i+1]
    print(f' fitting segment {i} in range {x1:.2f} to {x2:.2f}')
    # select points

    # broaden =0 is strict limit fitting
    # broaden > 0 widens the slection to catch any additional points that should be fit
    broaden = 0.5
    xlower = x1-broaden
    xupper = x2+broaden
    m = (tasg['MJD'] > xlower) * (tasg['MJD'] < xupper)
    npoi = m.sum()
    print(f'npoints {npoi}')
    if npoi > 2:
        print(f'fitting...')
        # calculate mean x position so that slopes are not insane...
        x = tasg['MJD'][m]
        meanx = np.mean(x)
        x = x - meanx
        y = tasg['fnorm'][m]
        yerr=tasg['fnormerr'][m]

        (m, merr, b, berr) = linfit_hogg(x,y,yerr)
        fit_x.append(meanx)
        fit_m.append(m)
        fit_merr.append(merr)
#        fit_width.append(xupper-xlower)
        fit_width.append(np.max(x)-np.min(x))

        # calculate max tau (just get minimum flux point, 1 - that...)
        max_tau.append(1.-np.min(y))

        # get the upper and lower points for the range we have, and calculate the fitted line values at those extrema so that we can plot it
        x_lower.append(np.min(x)+meanx)
        x_upper.append(np.max(x)+meanx)

        # calculate y values for these points
        y_lower.append(m*np.min(x)+b)
        y_upper.append(m*np.max(x)+b)

fit_x = np.array(fit_x)
fit_m = np.array(fit_m)
fit_merr = np.array(fit_merr)
fit_width = np.array(fit_width)
max_tau = np.array(max_tau)

x_lower = np.array(x_lower)
y_lower = np.array(y_lower)
x_upper = np.array(x_upper)
y_upper = np.array(y_upper)

print(x_lower)

# remove any with gradients larger than 1 as unphysical
m_lim = 1.0
print(f'removing any gradients with m > {m_lim} as that is a fit to one isolated point')
mb = (fit_m < m_lim)
fit_x = fit_x[mb]
fit_m = fit_m[mb]
fit_merr = fit_merr[mb]
fit_width = fit_width[mb]
max_tau = max_tau[mb]

vtrans = (((np.abs(fit_m)*np.pi/(2*max_tau))*(star_rad/(1*u.day)))).to(u.km/u.s) # eq. 4.5
vtrans_err = vtrans * (fit_merr/np.abs(fit_m))

mh = (fit_width < 2.)

fax2.errorbar(fit_x,np.abs(fit_m),yerr=fit_merr,fmt='b.',mec='blue')
#fax2.errorbar(fit_x,np.abs(fit_m),yerr=fit_merr,fmt='w.',mec='black',alpha=0.3)
#fax2.errorbar(fit_x[mh],np.abs(fit_m[mh]),yerr=fit_merr[mh],fmt='k.')

#fax3.errorbar(fit_x,vtrans.value,yerr=vtrans_err.value,fmt='wv',mec='black',alpha=0.3)
fax3.errorbar(fit_x,vtrans.value,yerr=vtrans_err.value,fmt='k.',mec='black')
#fax3.errorbar(fit_x[mh],vtrans.value[mh],yerr=vtrans_err.value[mh],fmt='kv')


fax.set_ylabel('Normalised flux')

fax3.set_xlabel('Epoch [MJD]')
fax2.set_ylabel('Gradient of flux [$d^{-1}$]')
fax3.set_ylabel('$v_{trans}$ [$km\ s^{-1}$]')
fax.set_xlim(59600,59750) 
fax2.set_ylim(-0.05,0.4)

# just the calcualted end points
#fax.scatter(x_lower,y_lower,marker='1',color='orange')
#fax.scatter(x_upper,y_upper,marker='2',color='yellow')


for (x1,x2,y1,y2) in zip(x_lower,x_upper,y_lower,y_upper):
    fax.plot((x1,x2),(y1,y2),color='blue',alpha=0.3,zorder=-5)


tyb = dict(color='black', weight='bold', fontsize=12)
fax.text(0.03, 0.9, 'a', ha='right', va='top', transform=fax.transAxes, **tyb)
fax2.text(0.03, 0.9, 'b', ha='right', va='top', transform=fax2.transAxes, **tyb)
fax3.text(0.03, 0.9, 'c', ha='right', va='top', transform=fax3.transAxes, **tyb)

plt.savefig(paths.figures / 'master_lightcurve_nature_with_zoom_3_panel.pdf')
