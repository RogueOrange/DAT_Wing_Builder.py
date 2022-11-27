import urllib.request
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.express as px
from time import perf_counter as pfc

start = pfc()
tip_url = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=prandtl-d-tip-ns"
root_url = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=sp4621hp-po"
span = 1500
segments = 5000
span_percents = []


def gen_from_URL(url):
    return np.genfromtxt(urllib.request.urlopen(url), skip_header=1, dtype="float").T


def coord_coupling(x, y, z):
    return pd.DataFrame(
        {
            "X": x,
            "Y": y,
            "Z": z
        }
    ).T


# kinda the same as spline
def equalizer(r, t, q=None, **kwargs):
    """
    :param r:1st Set of AF Coords
    :param t:2nd Set of AF Coords
    :param q:Amount of points (optional)
    :return: 2 sets of Coords with equal length & AF starting and ending at (1,0),(0,0)
    """
    ep = kwargs.get('setEndPoints', False)
    if q is None:
        q = 30
    # defines the number of pts/the smoothness
    u_new = np.linspace(0, 1, q)

    # root interp
    root_tck, u = splprep([r[0], r[1]], s=0)[:3]
    new_r = splev(u_new, root_tck)

    # tip interp
    tip_tck, u = splprep([t[0], t[1]], s=0)[:3]
    new_t = splev(u_new, tip_tck)

    # root start & finish normalised
    if ep:
        new_r = correct_ends(new_r)
        new_t = correct_ends(new_t)
    return new_r, new_t


def correct_ends(xy):  # sourcery skip: min-max-identity
    if (xy[0] > 1).sum() > 1:
        print("Scale Aifoils Down")
        # TODO: auto scale AF to fit
        raise ValueError('airfoil has too many points outside bounds xmax=1,xmin=0')
    xy[0] = [0 if x < 0 else 1 if x > 1 else x for x in xy[0]]
    xy[0][xy[0].index(min(xy[0]))] = 0  # Sets the min val of x to 0
    xy[1][0] = 0
    xy[1][-1] = 0
    return xy


def spline(x, y, sq):
    """
    :param x: Set of X-Coords
    :param y: Set of Y-Coords
    :param sq: Spline Quality
    :return: New set of points generated using Spline Interpolation
    """
    tck, u = splprep([x, y], s=0)[:3]
    u_new = np.linspace(0, 1, sq)
    x, y = splev(u_new, tck)

    y[0], y[-1] = 0, 0

    return x, y


def inter(r, t, sp):
    """
    :param r:Root Coords
    :param t:Tip Coords
    :param sp: Span Percent
    :return:
    """
    return t + (r - t) * sp / 100


def populate(r, t, s, n, sq, **kwargs):
    """
    :param sq: Spline Quality Final
    :param r:Root Coords
    :param t:Tip Coords
    :param s: Section Span
    :param n: Number of intervals
    :return:
    """
    # TODO: Speed this up, prior version was 10x faster
    incl_tip = kwargs.get('inclTip', False)
    foo = []
    span_pos = np.linspace(0, s, n)
    span_per = [100 - i * 100 / s for i in span_pos]
    for i in range(n):
        xyz = inter(r, t, span_per[n - 1 - i])

        xyz = coord_coupling(xyz[0], xyz[1], xyz[2])
        foo.append(xyz)
    # if kwargs.get('inclTip', False):
    #     new_t = spline(t[0],t[1],sq)
    #     foo.append(coord_coupling(new_t[0],new_t[1],100))
    #

    return pd.concat(foo, keys=span_per)

    # r = coord_coupling(r[0], r[1], 1)
    # t = coord_coupling(t[0], t[1], 5)


def initial(r_url, t_url,**kwargs):
    sq = kwargs.get('SplineQuality', 30)
    r = gen_from_URL(r_url)
    t = gen_from_URL(t_url)
    r, t = equalizer(r, t,sq)
    r, t = correct_ends(r), correct_ends(t)
    r.append([0 for _ in range(len(t[0]))])
    t.append([span for _ in range(len(t[0]))])
    return np.array(r), np.array(t)


def sweep(r, t, angle):
    """
    :param r: Root coords
    :param t: Tip coords
    :param angle: Sweep angle
    :return: New tip coords
    """
    # TODO: write function to apply sweep to the tip airfoil

    return r, t


# TODO: write dihedral method that accepts 2 sets of coordinates and an angle
#   and translates the 2nd set of points
#   (Only changes yz, z-points must remain || to xy plane )

# TODO: write twist method that applies a given twist to each section
#  (only changes xy, z-points must remain || to xy plane )

# TODO: write method to generate points from a
#  6th degree polynomial or any better function
#  that can describe a wide variety of twists

def writeFile(df,writePath):
    with open(writePath, 'w') as f:
        dfAsString = df.to_string(header=False, index=False)
        f.write(dfAsString)


root, tip = initial(root_url, tip_url,SplineQuality=200)

sections = populate(root, tip, span, segments, 200)
#save_location = "C:/Users/garet/OneDrive - KU Leuven/Desktop/Coding/Python/Passage.txt"
#writeFile(sections,save_location)
print(sections)
dur = pfc() - start
print(dur)



