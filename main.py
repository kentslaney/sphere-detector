import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from functools import partial, cached_property
import sympy as sy

local = pathlib.Path(__file__).parents[0]
mlmodel = None
target = np.array([392, 518])
diag = np.sqrt(np.sum(target ** 2))

def mlpackage():
    global Image
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()

    import coremltools as ct
    return ct.models.MLModel('DepthAnythingV2SmallF16.mlpackage')

def fs(pth, npy = None):
    global mlmodel
    if type(npy) is str:
        npy = pathlib.Path(npy)
    if npy is not None and npy.exists():
        return np.load(npy)
    if mlmodel is None:
        mlmodel = mlpackage()
    im = Image.open(pth)
    size = np.array(im.size)
    scaled = np.int32(np.max(target / size) * size)
    im_s = im.resize(scaled)
    origin = (scaled - target[::-1]) // 2
    im_t = im_s.crop(np.concat((origin, origin + target[::-1])))
    out = np.array(mlmodel.predict({"image": im_t})['depth'])
    if npy is not None:
        np.save(npy, out)
    return out

im4 = fs(local / "IMG_0004.HEIC", local / "out4.npy")
im5 = fs(local / "IMG_0005.HEIC", local / "out5.npy")

def grad(arr):
    return sobel(arr, 0) / 8, sobel(arr, 1) / 8

def hessian(d0, d1):
    return list(map(grad, [d0, d1]))

# https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
# L = T / 2 \pm (T ** 2 / 4 - D) ** 1/2
# λ[0] = ∂∥∂∥ <= ∂⟂∂⟂ = λ[1]
def eigenvalues2x2(sq):
    tr = sq[0][0] + sq[1][1]
    det = sq[0][0] * sq[1][1] - sq[0][1] * sq[1][0]
    center = tr / 2
    radius = np.sqrt(tr ** 2 / 4 - det)
    return [center - radius, center + radius]

def normals(d0, d1, t):
    t *= diag
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    e0, e1 = (c0 + d0 * t).flatten(), (c1 + d1 * t).flatten()
    return interpolate(e0, e1)

def interpolate(e0, e1):
    out = np.zeros(target)
    f0, f1 = np.int32(np.floor(e0)), np.int32(np.floor(e1))
    g0, g1 = e0 - f0, e1 - f1
    for i0, i1 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        z0, z1 = f0 + i0, f1 + i1
        y0, y1 = (1 - i0) + (2 * i0 - 1) * g0, (1 - i1) + (2 * i1 - 1) * g1
        x0 = np.logical_and(z0 >= 0, z0 < target[0])
        x1 = np.logical_and(z1 >= 0, z1 < target[1])
        w = np.logical_and(x0, x1)
        out[z0[w], z1[w]] += y0[w] * y1[w]
    return out

def casts(im, s0, s1, w):
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    delta = im - w
    e0, e1 = c0 + delta * s0, c1 + delta * s1
    return interpolate(e0[delta > 0].flatten(), e1[delta > 0].flatten())

def sources(d0, d1, t, s1, s0):
    t *= diag
    c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
    e0, e1 = c0 + d0 * t, c1 + d1 * t
    return np.log(np.sqrt((e0 - s0) ** 2 + (e1 - s1) ** 2))

def slide(f, **kw):
    frame = f(0)

    from matplotlib.widgets import Slider, Button
    fig, ax = plt.subplots()
    out = ax.imshow(frame, vmin=0, vmax=4)
    fig.subplots_adjust(bottom=0.25)

    def onclick(event):
        if event.inaxes != ax:
            return
        out.set_data(sources(d0, d1, slider.val, event.xdata, event.ydata))
        fig.canvas.draw_idle()
    # fig.canvas.mpl_connect('button_press_event', onclick)

    axt = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax=axt, label='t', **kw)
    def update(val):
        nonlocal frame
        frame = f(slider.val)
        out.set_data(frame)
        fig.canvas.draw_idle()
    slider.on_changed(update)

    bax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(bax, 'reset', hovercolor='0.975')
    def bev(event):
        out.set_data(frame)
        fig.canvas.draw_idle()
    button.on_clicked(bev)

    plt.show()

def slide0(arr):
    d0, d1 = grad(arr)
    return slide(partial(normals, d0, d1), valmin=0, valmax=8, valinit=0)

def adjust(arr):
    d0, d1 = grad(arr)
    sq = hessian(d0, d1)
    para, perp = eigenvalues2x2(sq)
    coef = np.divide(para, perp ** 3, out=np.ones_like(arr), where=perp != 0)
    return d0 * coef, d1 * coef

def slide1(arr):
    s0, s1 = adjust(arr)
    return slide(partial(casts, arr, s0, s1), valmin=0, valmax=1, valinit=0.1)

def density(arr, samples=100, scale=4):
    d0, d1 = grad(arr)
    t = np.linspace(0, scale, samples)
    out = np.zeros((samples,) + arr.shape)
    for i in range(samples):
        out[i] = normals(d0, d1, t[i])
    return out

def blur(arr, sigma=3):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)

def ndmax(arr):
    return np.unravel_index(arr.argmax(), arr.shape)

class Sphere:
    t0, t1 = target // 2
    offset = sy.Rational(1, 10)
    def __init__(self, r=np.min(target) // 5):
        self.r = r
        self.a, self.b = self.dim = sy.symbols('a b')

    @cached_property
    def rescale(self):
        return sy.Rational(4, 5 * self.r)

    @cached_property
    def im(self):
        out = np.zeros(target)
        c1, c0 = np.meshgrid(*map(np.arange, target[::-1]))
        d2 = (self.t0 - c0) ** 2 + (self.t1 - c1) ** 2
        out[d2 < self.r ** 2] = np.sqrt(self.r ** 2 - d2[d2 < self.r ** 2])
        out *= float(self.rescale)
        out[d2 < self.r ** 2] += float(self.offset)
        return out

    def sym(self):
        return self.offset + self.rescale * sy.sqrt(
                self.r ** 2 - (self.t0 - self.a) ** 2 - (self.t1 - self.b) ** 2)

    def grad(self):
        expr = self.sym()
        return sy.Matrix([[sy.diff(expr, self.a)], [sy.diff(expr, self.b)]])

    def hessian(self):
        expr = self.grad()
        return sy.Matrix.hstack(sy.diff(expr, self.a), sy.diff(expr, self.b))

    def eigenvalues(self):
        expr = self.hessian()
        tr = expr.trace()
        center = tr / 2
        radius = sy.sqrt(tr ** 2 / 4 - expr.det())
        return sy.Matrix([[center - radius], [center + radius]])

    def coef(self):
        para, perp = self.eigenvalues()
        return para / perp ** 3

    def adjusted(self):
        return self.coef() * self.grad()

    def at(self, *args):
        return list(zip(self.dim, args))

    def subs(self, a, b):
        return self.adjusted().subs(self.at(a, b))

    @cached_property
    def approx(self):
        return adjust(self.im)

    def cmp(self, a, b):
        return [i[a, b] for i in self.approx], \
                self.subs(a, b).T.evalf().tolist()[0]

def tmp0():
    slide1(Sphere().im)

# https://www.desmos.com/c/h0pkuzhfzh
# https://www.desmos.com/3d/9ijh5b7ok9
# m = ∂∥∂∥ / (∂⟂∂⟂)^3 * ∂∥ wrt gradient
# https://www.desmos.com/3d/pt8citup10

def tmp1(im):
    for i in im:
        plt.plot(i)
    plt.show()

def tmp2(delta0=-10, delta1=-10):
    mock = Sphere()
    ex = (mock.t0 + delta0, mock.t1 + delta1)
    print("outputs", mock.cmp(*ex))
    print("hessians")
    print([j[*ex] for i in hessian(*grad(mock.im)) for j in i])
    print(mock.hessian().subs(mock.at(*ex)).evalf())
    print("grad")
    print([j[*ex] for j in grad(mock.im)])
    print(mock.grad().subs(mock.at(*ex)).evalf())
    print("value")
    print(mock.im[*ex])
    print(mock.sym().subs(mock.at(*ex)).evalf())
    print("3x3 numerical")
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            print(mock.im[ex[0] + i, ex[1] + j], end=" ")
    print()

def tmp3():
    a, b, c, p, w = sy.symbols('a b c p w')
    expr = c + sy.sqrt(p ** 2 - a ** 2 - b ** 2) / w

    change = [(b, 0)]
    f = expr.subs(change)
    da = sy.simplify(sy.diff(expr, a).subs(change))
    da2 = sy.simplify(sy.diff(expr, a, a).subs(change))
    db2 = sy.simplify(sy.diff(expr, b, b).subs(change))

    print(sy.simplify((da2 / db2 - 1) / da ** 2))

if __name__ == "__main__":
    tmp3()
