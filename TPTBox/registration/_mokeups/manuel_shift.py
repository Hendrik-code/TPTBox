from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import NII

name = "spinegan0064/target-302_from-301/"

p = f"/media/data/robert/datasets/spinegan_T2w/test_nii/{name}/"
p2 = Path(p, "seg.nii.gz")
p = Path(p, "seg2.nii.gz")
# p = list(Path(p).glob("*-vert_msk.nii.gz"))
# assert len(p) == 1
# p = p[0]
nii = NII.load(p, True)

print(nii.orientation)
arr = nii.get_array()
assert nii.orientation[1] == "P"
assert nii.orientation[2] == "I"
assert nii.orientation[1] == "P"
out = np.zeros_like(arr)


def abs2(a):
    if a is None:
        return 0
    else:
        return abs(a)


def inv2(a):
    if a == 0:
        return None
    else:
        return -a


def shift(arr: np.ndarray, out: np.ndarray, id: int, P: int, I: int):
    arr2 = arr.copy()
    arr2[arr != id] = 0
    arr[arr == id] = 0

    if P >= 0:
        Ps = slice(P, None)
    else:
        Ps = slice(0, P)
    if I >= 0:
        Is = slice(I, None)
    else:
        Is = slice(0, I)
    Ps2 = slice(abs2(Ps.stop), inv2(Ps.start))
    Is2 = slice(abs2(Is.stop), inv2(Is.start))
    arr2[:, Ps2, Is2][out[:, Ps, Is] != 0] = 0
    out[:, Ps, Is] += arr2[:, Ps2, Is2]


def rm(arr: np.ndarray, id: int):
    arr[arr == id] = 0


def rest(arr, out):
    arr[out != 0] = 0
    out += arr


name = "spinegan0016/target-302_from-301/"
rm(arr, 15)
rest(arr, out)
nii.set_array(out, inplace=True)
nii.save(p2)


# name = "spinegan0016/target-302_from-301/"
# rm(arr, 14)
# rm(arr, 15)
# shift(arr, out, 16, 1, 0)
# shift(arr, out, 17, 1, 0)
# shift(arr, out, 18, 2, 0)
# shift(arr, out, 19, 2, 0)
# shift(arr, out, 20, 1, 0)
# shift(arr, out, 21, 1, 0)
#
# rm(arr, 24)
# rest(arr, out)
# nii.set_array(out, inplace=True)
# nii.save(p2)
#


# name = "/spinegan0043/target-301_from-206/"
# shift(arr, out, 1, 4, -2)
# shift(arr, out, 2, 1, 0)
# shift(arr, out, 3, 1, 0)
# shift(arr, out, 6, 0, 1)
# shift(arr, out, 7, 1, 0)
# shift(arr, out, 8, 0, -1)#

# rm(arr, 14)
# rest(arr, out)
# nii.set_array(out, inplace=True)
# nii.save(p2)
# name = "spinegan0052/target-301_from-207/"
# shift(arr, out, 9, 1, 1)
# shift(arr, out, 10, 1, 0)
# shift(arr, out, 11, 2, 0)
# shift(arr, out, 12, 2, 0)
# shift(arr, out, 13, 3, 0)
#
# rm(arr, 14)
# rest(arr, out)

# name = "spinegan0052/target-301_from-30299/"
# shift(arr, out, 1, 6, 0)
# shift(arr, out, 2, 3, 0)
# shift(arr, out, 3, 1, 0)
# shift(arr, out, 9, 3, 0)
# rm(arr, 10)
# rest(arr, out)
# name = "spinegan0052/target-302_from-207/"
# rm(arr, 12)
# shift(arr, out, 13, 1, 1)
# shift(arr, out, 14, 1, 0)
# shift(arr, out, 15, 1, 0)
# shift(arr, out, 16, 1, 0)
# shift(arr, out, 17, 2, 1)
# shift(arr, out, 18, 2, 1)
# shift(arr, out, 19, 3, 0)
# shift(arr, out, 20, 2, -1)
# rm(arr, 64512)
# rest(arr, out)
# name = "spinegan0060/target-301_from-20498/"#
# shift(arr, out, 18, 3, -1)
# shift(arr, out, 19, 4, 0)
# shift(arr, out, 20, 4, 0)
# shift(arr, out, 21, 4, 0)
# shift(arr, out, 22, 3, 0)
# shift(arr, out, 23, 2, 0)#

# rest(arr, out)
# name = "spinegan0064/target-301_from-301/"
# rm(arr, 17)
# shift(arr, out, 16, 1, 0)
# shift(arr, out, 15, 1, 0)
# shift(arr, out, 14, 1, 0)
# shift(arr, out, 13, 1, 0)
# shift(arr, out, 12, 1, 0)
# shift(arr, out, 14, 1, 0)#
# rest(arr, out)
# name = "spinegan0064/target-302_from-301/"
# rm(arr, 16)
# shift(arr, out, 17, 2, 0)
# shift(arr, out, 18, 2, 0)
# shift(arr, out, 19, 1, 0)
# shift(arr, out, 20, 1, 0)
# shift(arr, out, 21, 2, 0)#
# rest(arr, out)
# name = "spinegan0073/target-302_from-201/"
# shift(arr, out, 16, 2, 0)
# shift(arr, out, 17, 4, 0)
# shift(arr, out, 18, 4, 0)
# shift(arr, out, 19, 4, 1)
# shift(arr, out, 20, 3, 0)
# shift(arr, out, 21, 3, 0)
# shift(arr, out, 22, 2, 0)
# rm(arr, 23)
# rest(arr, out)
# name = "spinegan0077/target-302_from-303/"
# shift(arr, out, 17, 3, 0)
# shift(arr, out, 18, 3, 0)
# shift(arr, out, 19, 2, 0)
# shift(arr, out, 21, 1, 0)
# shift(arr, out, 22, 1, 0)
# shift(arr, out, 23, 0, 1)
# shift(arr, out, 24, 0, 1)
# shift(arr, out, 25, 0, 1)
#
# rest(arr, out)
# name = "spinegan0093/target-302_from-12/"
# rm(arr, 10)
# rm(arr, 11)
# shift(arr, out, 12, 2, -1)
# shift(arr, out, 13, 2, -1)
# shift(arr, out, 14, 3, -1)
# shift(arr, out, 15, 3, -1)
# shift(arr, out, 16, 3, -1)
# shift(arr, out, 17, 3, -1)
# shift(arr, out, 18, 3, -1)
# shift(arr, out, 19, 4, -1)
# shift(arr, out, 20, 2, -1)
# shift(arr, out, 21, 3, 0)
# rm(arr, 22)
# rest(arr, out)


# name = "spinegan0093/target-303_from-12/"
# rm(arr, 19)
# rm(arr, 20)
# shift(arr, out, 21, 3, 0)
# shift(arr, out, 22, 1, 0)
# shift(arr, out, 23, 1, 0)
# shift(arr, out, 24, 2, 2)#

# rest(arr, out)
# name = "spinegan0105/target-302_from-201/"
# shift(arr, out, 19, 4, 0)
# shift(arr, out, 20, 4, -1)
# shift(arr, out, 21, 3, 0)
# shift(arr, out, 22, 3, 0)
# shift(arr, out, 23, 1, 0)
# shift(arr, out, 24, 0, -1)#

# rest(arr, out)
# rm(arr, 7)
# name = "spinegan0113/target-301_from-None/"
##              P, I
# shift(arr, out, 9, 1, 0)
# shift(arr, out, 10, 1, 0)
# shift(arr, out, 11, 1, 0)
# shift(arr, out, 12, 2, -1)
# shift(arr, out, 13, 2, -1)
# shift(arr, out, 14, 2, -1)
# shift(arr, out, 15, 2, -1)
# shift(arr, out, 16, 2, -1)
# shift(arr, out, 17, 3, -1)
# rm(arr, 18)
# rest(arr, out)
