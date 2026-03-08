from .display import Example, Readable
from .utils import local, examples

cache = local / "cache"

im4 = Example.file(examples / "IMG_0004.HEIC", cache / "da2_4.npy", "im4")
im5 = Example.file(examples / "IMG_0005.HEIC", cache / "da2_5.npy", "im5")
im7 = Example.file(examples / "IMG_0007.HEIC", cache / "da2_7.npy", "im7")
im8 = Example.file(examples / "IMG_0008.HEIC", cache / "da2_8.npy", "im8")

