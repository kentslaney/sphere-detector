import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabulate import tabulate, SEPARATING_LINE

from .detect import Raster

def poplt(x=None, init=None):
    if x is not None:
        return x, x.gca()
    fig = plt.figure()
    ax = fig.subplots()
    if init is not None:
        init(ax)
    return fig, ax

class Wrapper:
    def __init__(self, wrapping, name=None):
        self.obj, self.name = wrapping, name

    def __getattr__(self, name):
        return getattr(self.obj, name)

class Example(Wrapper):
    @classmethod
    def file(cls, path, npy=None, name=None, **kw):
        obj = cls(Raster.file(path, npy, **kw))
        obj.name = pathlib.Path(path).name if name is None else name
        return obj

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
                    str(i), x[1::-1], xytext=(1, -1),
                    textcoords="offset points",
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

    def opt(self, *a, **kw):
        obj = Sampled(self.obj.opt(*a, **kw), self.name)
        return obj

class Sampled(Wrapper):
    def plot_depths(self, index=0, ax=None):
        y = jnp.concat(self.adjacent, axis=1) * self.w[:, None, None]
        y += self.surface.skew
        y_c, rmse = self.surface.center_2nd * self.w, self.surface.rmse

        if ax is None:
            fig = plt.figure()
            ax = fig.subplots()

        cmap, taus = plt.get_cmap('twilight'), self.surface.revolutions
        for i in range(0, self.samples.shape[1], 1):
            ax.plot(y[index][i][:self.samples[index][i]], color=cmap(taus[i]))

        ax.axvline(x=self.fit.radius[index], color='g')
        import matplotlib.patches as patches
        kw = {
                'linewidth': 2, 'edgecolor': 'r', 'facecolor': 'none',
                'zorder': 5 }
        ax.add_patch(patches.Circle(
            (self.surface.x_c, y_c[index]), self.fit.radius[index], **kw))

        name = " " if self.name is None else f" {self.name} "
        msg = (
            f'Surface RMSE: {rmse[index]:.2f}\n'
            f'Edge RMSE: {self.fit.rmse[index]:0.2f}\n'
            f'n: {self.fit.samples[index]} / {self.surface.config.rays}'
        )
        ax.text(
                0.05, 0.05, msg,
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.set_title(f"Surface Depths for{name}Sphere Candidate {index}")
        ax.set_xlabel("Distance from Initial Center (pixels)")
        ax.set_ylabel("Slice Depth (pixels)")

        if ax is None:
            return fig

    def tsv(self, residuals, index=0):
        yx = jnp.concat(self.steps, axis=2)
        center = jnp.mean(yx, (2, 3), where=self.valid, keepdims=True)
        data = jnp.concat((yx - center, residuals[None]))
        unsized = data[:, index][:, self.valid[index]].T
        return "\n".join(["\t".join(str(j.item()) for j in i) for i in unsized])

    @property
    def surface(self):
        return Candidate(self.obj.surface, self.name)

class Candidate(Wrapper):
    def debug(self):
        res = []
        for i in jnp.nonzero(self.edge.valid)[0]:
            res.append([j.item() if hasattr(j, "item") else j for j in [
                i,
                self.edge.center_1st[i],
                self.edge.center_0th[i],
                self.edge.radius[i],
                self.edge.samples[i],
                self.edge.rmse[i],
                self.rmse[i],
                self.confidence[i],
            ]])
        if self.name is not None:
            res[0] = [self.name] + res[0]
            for i in range(1, len(res)):
                res[i] = [""] + res[i]
        return res

class Readable:
    headers = ("i", "x", "y", "r", "n", "edge", "surface")

    def __init__(self, it):
        self.value = []
        for i in it:
            self.value += i
            self.value.append(SEPARATING_LINE)
        self.value = self.value[:-1]

    def __repr__(self):
        headers = ((
            ("name",) if len(self.headers) < len(self.value[0]) and \
                    isinstance(self.value[0][0], str) else ()) + self.headers)
        headers = headers + ("",) * max(0, len(self.value[0]) - len(headers))
        return tabulate(self.value, headers=headers)
