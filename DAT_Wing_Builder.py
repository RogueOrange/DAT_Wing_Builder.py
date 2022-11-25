import urllib.request
import numpy as np
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
import plotly.express as px

np.set_printoptions(formatter={'float': '{:0.3f}'.format})

interval_count = 800
interval_step = 50


def gen_url_pts(url):
    """

    :param url: URL of airfoil -from AirfoilTools.com
    :return: List of XY coordinates defining the airfoil
    """
    return np.genfromtxt(urllib.request.urlopen(url), skip_header=1, dtype="float").T


def equalizer(r, t):
    """
    :param r: First Array [x,y]
    :param t: Second Array [x,y]
    :return: Generates 2 equally sized arrays using spline interpolation
    """
    pts = 30

    u_new = np.linspace(0, 1, pts)

    # r interp
    root_tck, u = splprep([r[0], r[1]], s=0)[:3]
    new_r = splev(u_new, root_tck)

    # t interp
    tip_tck, u = splprep([t[0], t[1]], s=0)[:3]
    new_t = splev(u_new, tip_tck)

    return new_r, new_t


class Section:
    def __init__(self, tc: float, rc: float, hs: float, splq: int, tu: str, ru: str, swp: float = None):
        """

        :param tc: Tip Chord
        :param rc: Root Chord
        :param hs: Half Span
        :param splq: Spline Quality
        :param tu: Tip URL
        :param ru: Root URL
        :param swp: Sweep
        """
        self.root_url = ru
        self.tip_url = tu
        self.spline_quality = splq
        self.tip_chord = tc
        self.root_chord = rc
        self.hs = hs
        self.sweep = swp
        self.root, self.tip = self.rt_scale()

        self.span_positions = self.span_pos(interval_count, interval_step)
        self.span_percents = self.span_percents()
        self.afs = self.create_foils()
        self.afs = self.wing_twister()
        self.sweeper()

    def mirror(self):
        mrd_pts = np.flipud(self.afs)
        copy_afs = np.concatenate((self.afs, mrd_pts))
        return copy_afs

    def rt_scale(self):
        r1 = gen_url_pts(self.root_url)
        t1 = gen_url_pts(self.tip_url)
        r2, t2 = equalizer(r1, t1)
        nr, nt = np.array(list(scale(self.root_chord, np.array(r2)))), \
                 np.array(list(scale(self.tip_chord, np.array(t2))))
        return nr, nt

    def span_pos(self, ic=None, isp=None):
        """
        :param ic: Amount of Intervals
        :param isp: interval Step
        :return: Span Positions i.e. Array of Coordinates evenly distributed along span
        """
        return np.linspace(0, self.hs, ic) if ic else np.arange(0, self.hs, isp)

    def span_percents(self):
        """
        :return: List of percentages of span position to half span
        """
        return [(100 - sp * 100 / self.hs) for sp in self.span_positions]

    def create_foils(self):
        """
        :return: Array of afs between root and tip
        """
        return np.array([inter(self.root, self.tip, foo) for foo in self.span_percents])

    def sweeper(self):
        """
        :return: Coordinates translated to account for sweep
        """
        x_loc = np.linspace(0, self.hs * np.tan(np.deg2rad(self.sweep)), len(self.afs))

        for foo in range(interval_count):
            self.afs[foo][0] = list(map(lambda bar: bar + x_loc[foo], self.afs[foo][0]))
        pass

    def plot_af(self, clrs=None):
        """
        :param clrs: list of colors for wing colorscale
        :return: Plots Airfoils
        """

        if clrs is None:
            clrs = list(["#FF4F00", "#00407A"])

        # making colorscale
        color_scale_length = np.linspace(0, 1, len(self.span_positions))

        unique_color_scale = px.colors.make_colorscale(clrs)
        color_scale = px.colors.sample_colorscale(unique_color_scale, list(color_scale_length))

        color_scale[0] = color_scale[-1]
        color_scale[-1] = color_scale[1]
        for i, af in enumerate(self.afs):
            # print(af[0], af[1], np.full(len(af[0]), self.span_positions[i]), sep="\n", end="\n\n\n")
            x, y = spline(af[0], af[1], self.spline_quality)
            z = np.full(self.spline_quality, self.span_positions[i])

            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       name=str(round((100 - self.span_percents[i]), 2)) + "%",
                                       mode='lines',
                                       line_color=color_scale[i],
                                       line=dict(width=3)))

            # fig.update_traces(line={'size': 10})

            # if mirror is True:
        for i, af in enumerate(self.afs):
            # print(af[0], af[1], np.full(len(af[0]), self.span_positions[i]), sep="\n", end="\n\n\n")
            x, y = spline(af[0], af[1], self.spline_quality)
            z = np.full(self.spline_quality, self.span_positions[i])

            fig.add_trace(go.Scatter3d(x=x, y=y, z=-z,
                                       name=str(round((100 - self.span_percents[i]), 2)) + "%",
                                       mode='lines',
                                       line_color=color_scale[i],
                                       line=dict(width=5)))

            # fig.update_traces(line={'size': 10})

        fig.update_layout(legend=dict(y=1.0, traceorder='reversed', font_size=16),
                          scene=dict(
                              aspectratio=dict(
                                  x=5, y=1, z=20),
                              xaxis=dict(range=[-15, 75]),
                              yaxis=dict(range=[-18, 18]),
                              zaxis=dict(range=[-180, 180])

                          ))
        # if mirror is True:
        fig.show()

    def wing_twister(self, **kwargs):
        """

        :param kwargs:-Twist_distribution i.e. type of twist.
                       :-Twist angle in radians

        :return: Rotated Airfoils
        """
        max_twist = None
        if max_twist is None:
            max_twist = 0.4
        twist = np.geomspace(max_twist, 1, len(self.span_positions)) - 0.8

        return [rotation(angle, xy) for angle, xy in zip(twist, self.afs)]


def inter(r, t, sp):
    return t + (r - t) * sp / 100


def plot(pts, clrs=None):
    """

    :param pts:[[x][y][z]]
    :param clrs: Colors used to create the wing color contour
    :return: Plots the wing
    """
    # making colorscale
    color_scale_length = np.linspace(0, 1, len(pts[0]))
    unique_color_scale = px.colors.make_colorscale(clrs)
    color_scale = px.colors.sample_colorscale(unique_color_scale, list(color_scale_length))
    color_scale[0] = color_scale[-1]

    for i, af in enumerate(pts):
        # print(af[0], af[1], np.full(len(af[0]), self.span_positions[i]), sep="\n", end="\n\n\n")
        x, y, z = pts
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode='lines',
                                   line_color=color_scale[i],
                                   line=dict(width=10)))

        # fig.update_traces(line={'size': 10})

        fig.update_layout(legend=dict(y=1.0, traceorder='reversed', font_size=16),
                          scene=dict(
                              aspectratio=dict(
                                  x=5, y=1, z=10),
                              xaxis=dict(range=[-15, 75]),
                              yaxis=dict(range=[-18, 18]),
                              zaxis=dict(range=[0, 180])

                          ))
    fig.show()
    pass


def spline(x, y, quality):
    """

    :param x: X-Coordinates
    :param y: Y-Coordinates
    :param quality: Spline Quality
    :return: New set of XY points
    """
    tck, u = splprep([x, y], s=0)[:3]
    u_new = np.linspace(0, 1, quality)
    x, y = splev(u_new, tck)
    return x, y


def scale(scale_factor, pts):
    """
    :param scale_factor: Enough Said
    :param pts: List of coordinates
    :return: Scaled list of coordinates
    """
    return map(lambda x: x * scale_factor, pts)


def rotation(alfa, array):
    """
    :param alfa: Rotation Angle
    :param array: Array to rotate
    :return: Rotated Array
    """
    rot = np.array([[np.cos(alfa), -np.sin(alfa)], [np.sin(alfa), np.cos(alfa)]])
    translation1 = np.array([[1, 0, -array[0][0]], [0, 1, -array[1][0]], [0, 0, 1]])
    translation2 = np.array([[1, 0, array[0][0]], [0, 1, array[1][0]], [0, 0, 1]])
    array = np.array([array[0], array[1], len(array[1]) * [1]])
    array = np.matmul(translation1, array)
    array = np.delete(array, 2, 0)
    array = np.matmul(rot, array)
    row = len(array[1]) * [1]
    array = np.vstack([array, row])
    array = np.matmul(translation2, array)
    array = np.delete(array, 2, 0)
    return array


tip_chord = 5
root_chord = 15
half_span = 50
af_pts = 50
sweep = 40
spline_quality = 5
y_range = int(root_chord / 2)
tip_url = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=goe244-il"
root_url = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=marsden-il"

fig = go.Figure()
span = Section(tip_chord,root_chord,half_span, 40, tip_url, root_url, 12)
span.plot_af()
