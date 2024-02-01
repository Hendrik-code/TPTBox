from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
import numpy as np

from TPTBox import NII, Centroids, Log_Type, No_Logger, calc_centroids_from_subreg_vert, to_nii
from TPTBox.registration.ridged_points.mask_fixed import mask_fix_
from TPTBox.registration.ridged_points.reg_segmentation import ridged_from_points
from TPTBox.snapshot2D import ct_mri_snapshot

logger = No_Logger()
from TPTBox import BIDS_Global_info, Subject_Container


def dice(img1: NII, img2: NII, threshold=0.5, empty_score=0.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
    Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
    Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
    Dice coefficient as a float on range [0,1].
    Maximum similarity = 1
    No similarity = 0
    Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    im1 = img1.threshold(threshold).get_seg_array().astype(np.uint8)
    im2 = img2.threshold(threshold).get_seg_array().astype(np.uint8)
    if im1.shape != im2.shape:
        raise ValueError(f"Shape mismatch: im1 and im2 must have the same shape. {im1.shape} != {im2.shape}")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / im_sum


def evaluate_points(poi_T2w: Centroids, poi_ct: Centroids, keys_in, vert_t2w_nii, vert_ct_nii):
    transform = ridged_from_points(
        poi_T2w.filter_by_ids(keys_in),
        poi_ct.filter_by_ids(keys_in),
        representative_fixed=vert_t2w_nii,
        representative_movig=vert_ct_nii,
        verbose=False,
    )
    # crop = transform.compute_crop()
    vert_ct_nii_moved = transform.transform_nii(vert_ct_nii)
    del transform
    return dice(vert_ct_nii_moved, vert_t2w_nii)


def run_iterative_seg_on_ds(
    datasets=None,
    parents=None,
    formats="dixon",
    zooms=None,
    ax_code=None,
    n_jobs=16,
    filter_sub="",
):
    if parents is None:
        parents = ["rawdata", "derivatives"]
    if datasets is None:
        datasets = ["D:/data/dataset-spineGAN/"]
    bids_gl = BIDS_Global_info(
        datasets=datasets,
        parents=parents,
    )
    from joblib import Parallel, delayed

    a = Parallel(n_jobs=n_jobs)(
        delayed(__run_iterative_seg_help)(subj, formats, zooms, ax_code, filter_sub) for _, subj in bids_gl.enumerate_subjects()
    )
    print(a)


def __run_iterative_seg_help(subj: Subject_Container, formats, zooms: None | tuple[float, float, float], ax_code, filter_sub):
    q_ct = subj.new_query()
    q_ct.filter_format("ct")
    q_ct.filter("seg", "subreg")
    q_ct.filter("seg", "vert")
    q_ct.filter("sub", lambda x: filter_sub in x) if filter_sub != "" else None
    q_mr = subj.new_query()
    q_mr.filter_format(formats)
    q_mr.filter("seg", "subreg")
    q_mr.filter("seg", "vert")
    for ct_fam in q_ct.loop_dict():
        for mr_fam in q_mr.loop_dict(key_addendum=["part"]):
            try:
                # print(ct_fam)
                # print(mr_fam)
                try:
                    vert_t2w = mr_fam["msk_seg-vert_part-inphase"][0]
                    subreg_t2w = mr_fam["msk_seg-subreg_part-inphase"][0]
                    t2w = mr_fam["dixon_part-inphase"][0]
                except Exception:
                    vert_t2w = mr_fam["msk_seg-vert"][0]
                    subreg_t2w = mr_fam["msk_seg-subreg"][0]
                    t2w = mr_fam["dixon"][0]
                    # print(mr_fam)
                    # raise e
                vert_ct = ct_fam["msk_seg-vert"][0]
                subreg_ct = ct_fam["msk_seg-subreg"][0]
                ct = ct_fam["ct"][0]
                # POI
                poi_ct = subreg_ct.get_changed_path("json", format="poi")
                poi_T2w = subreg_t2w.get_changed_path("json", format="poi")
                #
                info = {"res": t2w.get("sequ")}
                path = str(Path(Path(ct.get_path_decomposed()[2]).parent, f'mr-{t2w.get("sequ")}_ct-{ct.get("sequ")}'))
                parent = "registration"
                if zooms is not None and zooms != (-1, -1, -1):
                    if zooms[0] == zooms[1] == zooms[2]:
                        parent += str(f"_iso_{zooms[0]:.1f}").replace(".", "_")
                    else:
                        parent += str(f"_({zooms[0]:.1f}-{zooms[1]:.1f}-{zooms[2]:.1f}").replace(".", "_")
                else:
                    zooms = (-1, -1, -1)
                vert_t2w_out = vert_t2w.get_changed_path("nii.gz", parent=parent, path=path, make_parent=False)
                subreg_t2w_out = subreg_t2w.get_changed_path("nii.gz", parent=parent, path=path, make_parent=False)
                vert_ct_out = vert_ct.get_changed_path("nii.gz", parent=parent, info=info, path=path, make_parent=False)
                subreg_ct_out = subreg_ct.get_changed_path("nii.gz", parent=parent, info=info, path=path, make_parent=False)
                ct_out = ct.get_changed_path("nii.gz", parent=parent, info=info, path=path, make_parent=False)
                t2w_out = t2w.get_changed_path("nii.gz", parent=parent, path=path, make_parent=False)
                snapshot = Path(t2w.dataset, parent, "snapshot", f'sub-{t2w.get("sub")}_mr-{t2w.get("sequ")}_ct-{ct.get("sequ")}.jpg')
                snapshot.parent.parent.mkdir(exist_ok=True)
                snapshot.parent.mkdir(exist_ok=True)
                if t2w_out.exists() and vert_t2w_out.exists() and subreg_t2w_out.exists():
                    ct_mri_snapshot(
                        t2w_out,
                        ct_out,
                        vert_ct_out,
                        # vert_t2w_out,
                        None,
                        # vert_ct_out,
                        vert_t2w_out,
                        None,
                        snapshot,
                    )  # if not snapshot.exists() else None
                    dice_v = dice(to_nii(vert_t2w_out), to_nii(vert_ct_out))
                    logger.print("exits", snapshot.name, f"{dice_v:.3}", ltype=Log_Type.OK)
                    return dice_v
                else:
                    logger.print("not exits", snapshot.name, ltype=Log_Type.NEUTRAL)
                    # continue
                logger.print("out", ct_out, ltype=Log_Type.NEUTRAL)
                ### Compute POI ###
                if not poi_T2w.exists():
                    poi = calc_centroids_from_subreg_vert(vert_t2w, subreg_t2w, decimals=4, subreg_id=list(range(40, 51)), verbose=True)
                    poi.save(poi_T2w, save_hint=2)
                if not poi_ct.exists():
                    poi = calc_centroids_from_subreg_vert(vert_ct, subreg_ct, decimals=4, subreg_id=list(range(40, 51)), verbose=True)
                    poi.save(poi_ct, save_hint=2)

                ### Select Points ###
                poi_T2w = Centroids().load(poi_T2w)
                poi_ct = Centroids().load(poi_ct)
                poi_ct_iso = poi_ct.rescale()
                poi_T2w_iso = poi_T2w.rescale()
                for key in poi_ct.centroids.copy():
                    if key not in poi_T2w:
                        # print("del key", key // 256, key % 256)
                        del poi_ct.centroids[key]
                for key in poi_T2w.centroids.copy():
                    if key not in poi_ct:
                        # print("del key", key // 256, key % 256)
                        del poi_T2w.centroids[key]
                from math import sqrt

                if len(poi_T2w) <= 2:
                    if ct_out.parent.exists() and len(list(ct_out.parent.iterdir())) == 0:
                        ct_out.parent.rmdir()
                    continue
                # print(poi_T2w.filter_by_subregion(50))
                poi_T2w_body = poi_T2w.filter_by_subregion(50)
                bets_body = []
                for key, key2 in zip(list(poi_T2w_body), list(poi_T2w_body)[1:], strict=False):
                    x, y, z = poi_T2w_iso[key]
                    x2, y2, z2 = poi_T2w_iso[key2]
                    l = sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
                    x, y, z = poi_ct_iso[key]
                    x2, y2, z2 = poi_ct_iso[key2]
                    l2 = sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
                    lmax = max(l, l2)
                    lmin = min(l, l2)
                    # print(f"{lmax / lmin:.3f}, {l:.1f}, {l2:.1f}, {abs(l-l2):.1f}")
                    bets_body.append((lmax / lmin, key))

                keys_in = [i[1] for i in sorted(bets_body)[:5]]
                keys_unused = [key for key in poi_ct.keys() if key not in keys_in]
                # keys_unused = [key for key in poi_ct.filter_by_subregion(50).keys() if key not in keys_in]
                keys_out = []

                vert_t2w_nii = to_nii(vert_t2w, True).clamp(max=1)
                vert_ct_nii = to_nii(vert_ct, True).clamp(max=1)
                if ax_code is not None:
                    vert_t2w_nii.reorient_(ax_code, verbose=True)
                    poi_T2w.reorient_(ax_code)
                if zooms != (-1, -1, -1) and zooms is not None:
                    vert_t2w_nii.rescale_(verbose=True)
                    poi_T2w.rescale_(zooms)

                if len(keys_in) <= 1:
                    # print(bets_body)
                    if ct_out.parent.exists() and len(list(ct_out.parent.iterdir())) == 0:
                        try:
                            ct_out.parent.rmdir()
                        except Exception:
                            pass
                    continue

                best_dice = evaluate_points(poi_T2w, poi_ct, keys_in, vert_t2w_nii, vert_ct_nii)
                ## Add points
                for key in keys_unused:
                    print(f"{best_dice:.4f}", key // 256, key % 256, "                    ", end="\r")
                    keys_in.append(key)

                    new_dice = evaluate_points(poi_T2w, poi_ct, keys_in, vert_t2w_nii, vert_ct_nii)
                    if new_dice >= best_dice:
                        best_dice = new_dice

                    else:
                        keys_in.pop()
                ## Remove Points
                for key in keys_in.copy():
                    print(f"{best_dice:.4f}", key // 256, key % 256, "remove?", "    ", end="\r")
                    keys_in.remove(key)

                    new_dice = evaluate_points(poi_T2w, poi_ct, keys_in, vert_t2w_nii, vert_ct_nii)
                    if new_dice >= best_dice:
                        best_dice = new_dice
                    else:
                        keys_in.append(key)
                print(best_dice)
                print([(i // 256, i % 256) for i in keys_in])
                transform = ridged_from_points(
                    poi_T2w.filter_by_ids(keys_in),
                    poi_ct.filter_by_ids(keys_in),
                    representative_fixed=vert_t2w_nii + 100,  # Replace with image
                    representative_movig=vert_ct_nii + 100,  # Replace with image
                    verbose=True,
                    ax_code=ax_code,
                    zooms=zooms,
                )
                t2w_nii = NII.load_bids(t2w).reorient(ax_code).rescale(zooms).clamp(min=0)
                ct_nii: NII = transform.transform_nii(ct).clamp(-1024, 1024)
                t2w_nii = t2w_nii.pad_to(ct_nii.shape)
                ct_nii.c_val = -1000
                crop = t2w_nii.compute_crop()
                crop = ct_nii.compute_crop(other_crop=crop)
                assert str(t2w_nii) == str(ct_nii), (str(t2w_nii), str(ct_nii))
                print(crop, t2w_nii, ct_nii)
                t2w_nii.apply_crop_(crop)
                ct_nii.apply_crop_(crop)
                # (slice(5, 166, None), slice(9, 280, None), slice(0, 49, None))
                ct_nii.save(ct_out, make_parents=True)
                mask_fix_(t2w_nii, ct_nii)
                t2w_nii.save(t2w_out, make_parents=True)

                vert = NII.load_bids(vert_t2w).reorient(ax_code).rescale(zooms).apply_crop_(crop)
                vert.pad_to(ct_nii.shape).save(vert_t2w_out, make_parents=True)
                subreg = NII.load_bids(subreg_t2w).reorient(ax_code).rescale(zooms).apply_crop_(crop)
                subreg.pad_to(ct_nii.shape).save(subreg_t2w_out, make_parents=True)
                transform.transform_nii(vert_ct).apply_crop_(crop).save(vert_ct_out, make_parents=True)
                transform.transform_nii(subreg_ct).apply_crop_(crop).save(subreg_ct_out, make_parents=True)

                ct_mri_snapshot(t2w_out, ct_out, vert_t2w_out, None, vert_ct_out, None, snapshot)
            except Exception:
                logger.print_error()


if __name__ == "__main__":
    # print(sys.argv)
    run_iterative_seg_on_ds(
        ["/media/data/robert/datasets/dataset-spinegan_T2w_all/"],
        n_jobs=1,
        filter_sub=sys.argv[1] if len(sys.argv) != 1 else "",
        ax_code=("R", "I", "P"),
        zooms=(1, 1, 1),
    )
