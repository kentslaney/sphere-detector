import sys, pathlib, grain
local = pathlib.Path(__file__).parents[0]

sys.path.insert(0, str(local))
from static import *
from depth12 import read_balls_depth_records
sys.path.pop(0)

ds = read_balls_depth_records()

def data_fields_only(ex):
    return {k: v for k, v in ex.items() if k != "file_name"}

ds = ds.map(data_fields_only)

@partial(
        jax.tree_util.register_dataclass,
        data_fields=["image", "depth", "label", "bbox"],
        meta_fields=[])
@dataclass
class BoundedBall(object):
    image: any
    depth: any
    label: any
    bbox: any

    def draw_annotations(self, ax):
        import matplotlib.patches as patches
        kw = { 'linewidth': 1, 'edgecolor': 'b', 'facecolor': 'none' }
        for i in jnp.unstack(self.bbox):
            if jnp.any(i == -1):
                continue
            rect = patches.Rectangle(i[:2], *(i[2:] - i[:2]), **kw)
            ax.add_patch(rect)
        return ax

    def raster(self):
        return Raster(self.image, self.depth)

def to_jax(ds): # TODO: switch to array_record and grain
    for i in ds:
        yield BoundedBall(**{k: jnp.array(v) for k, v in i.items()})

ds = to_jax(ds)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for n, i in enumerate(ds):
        ax = plt.subplots()[1]
        i.raster().draw_candidates(ax)
        i.draw_annotations(ax)
        plt.show()
        if (n + 1) % 100 == 0:
            input(f"shown {n + 1} of ~10k; press enter to continue:")
