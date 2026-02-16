import matplotlib.pyplot as plt

import sys, pathlib
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from detect import *
sys.path.pop(0)

def poplt(x=None, init=None):
    if x is not None:
        return x, x.gca()
    fig = plt.figure()
    ax = fig.subplots()
    if init is not None:
        init(ax)
    return fig, ax

class Example:
    obj = Raster
    name = None

    def __init__(self, wrapping):
        self.obj = wrapping

    @classmethod
    def file(cls, path, npy=None, name=None, **kw):
        obj = cls(cls.obj.file(path, npy, **kw))
        obj.name = pathlib.Path(path).name if name is None else name
        return obj

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def cropped_background(self, fig=None):
        return poplt(fig, lambda ax: ax.imshow(self.cropped()))

    def draw_sifted(self, fig=None, color='r'):
        fig, ax = self.cropped_background(fig)
        import matplotlib.patches as patches
        _, bboxes = self.seives.bound(self.config.candidates)
        kw = { 'linewidth': 1, 'edgecolor': color, 'facecolor': 'none' }
        for i, x in enumerate(jnp.unstack(bboxes)):
            rect = patches.Rectangle(x[1::-1], *(x[:1:-1] - x[1::-1]), **kw)
            ax.add_patch(rect)
            ax.annotate(
                    str(i), x[1::-1], xytext=(1, -1), textcoords="offset points",
                    va='top', ha='left', color=color)
        return fig

    def draw_centers(self, fig=None):
        fig, ax = self.cropped_background(fig)
        pred = self.stat(self.config.candidates)
        ax.scatter(*pred.mean.centers.T[::-1], color='b')
        for i, (y, x) in enumerate(pred.mean.centers):
            ax.annotate(str(i), (x, y), color='b')
        return fig

    def draw_refit(self, fig=None, label=True):
        fig, ax = self.cropped_background(fig)
        import matplotlib.patches as patches
        stats = self.opt(self.config.candidates).fit
        color = 'r'
        kw = { 'linewidth': 1, 'edgecolor': color, 'facecolor': 'none' }
        if label is not None:
            thetas = jax.random.uniform(
                    jax.random.key(label),
                    stats.radius.shape, maxval=2 * jnp.pi)
        it = jnp.unstack(
                jnp.stack((stats.center_0th, stats.center_1st, stats.radius)),
                axis=1)
        for y, x, r in it:
            ax.add_patch(patches.Circle((x, y), r, **kw))
        if label is not None:
            ax.autoscale(False)
            for i, (y, x, r) in enumerate(it):
                ax.plot(
                        [x, x + jnp.cos(thetas[i]) * r],
                        [y, y + jnp.sin(thetas[i]) * r], color=color)
                ax.annotate(str(i), (x, y), color=color)
        return fig

    def draw_candidate(self, index=0, fig=None):
        fig, ax = self.cropped_background(fig)
        opt = self.opt(index + 1)
        ax.scatter(*opt.origin[-1].T[::-1], color='r')
        ax.scatter(*opt.unsized[-1][::-1], color='b')
        return fig

    def plot_rays(self, index=0):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax0, ax1 = fig.subplots(2, 1)

        opt = self.opt(index + 1)
        for side in opt.adjacent:
            for cast in side[-1]:
                ax0.plot(cast)
        ax0.axhline(opt.depth_mean[0])
        ax0.axhline(opt.depth_mean[0] + self.config.alpha * opt.depth_std[0])
        ax0.axhline(opt.depth_mean[0] - self.config.beta * opt.depth_std[0])
        ax0.axvline(opt.radius_mean[0] + self.config.chi * opt.radius_std[0])

        for side in opt.adjacent:
            for cast in side[-1]:
                ax1.plot(cast[1:] - cast[:-1])
        ax1.axhline(-self.config.delta * opt.depth_std[0])
        ax1.axvline(opt.radius_mean[0] + self.config.chi * opt.radius_std[0])
        return fig

    def plot_depths(self, index=0):
        this = self.opt()
        y, (_, _, y_c, rmse) = jnp.concat(this.adjacent, axis=1), this.surface
        y, y_c = y * this.w[:, None, None], y_c * this.w

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.subplots()
        for i in range(this.samples.shape[1]):
            ax.plot(y[index][i][:this.samples[index][i]], color='b')

        ax.axvline(x=this.fit.radius[index], color='g')
        import matplotlib.patches as patches
        kw = { 'linewidth': 2, 'edgecolor': 'r', 'facecolor': 'none' }
        ax.add_patch(patches.Circle(
            (0, y_c[index]), this.fit.radius[index], zorder=5, **kw))

        name = " " if self.name is None else f" {self.name} "
        msg = (
            f'Surface RMSE: {rmse[index]:.2f}\n'
            f'Edge RMSE: {this.fit.rmse[index]:0.2f}\n'
            f'n: {this.fit.samples[index]} / {this.surface.config.rays}'
        )
        ax.text(
                0.05, 0.05, msg,
                transform=plt.gca().transAxes,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.set_title(f"Surface Depths for{name}Sphere Candidate {index}")
        ax.set_xlabel("Distance from Initial Center (pixels)")
        ax.set_ylabel("Slice Depth (pixels)")

        return fig

    def readable(self):
        this = self.opt().surface
        for i in jnp.nonzero(this.edge.valid)[0]:
            print(
                    f"[{i:2d}] "
                    f"({this.edge.center_1st[i]:7.1f}, "
                    f"{this.edge.center_0th[i]:7.1f}) "
                    f"r: {this.edge.radius[i]:7.2f} "
                    f"n: {this.edge.samples[i]:2d} "
                    f"edge: {this.edge.rmse[i]:7.3f} "
                    f"surface: {this.rmse[i]:7.3f} ")
        print("----")

