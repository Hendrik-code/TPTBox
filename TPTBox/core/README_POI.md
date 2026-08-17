# TPTBox `poi.py` — Points of Interest

![Example of two lumbar vertebrae. The left example is derived from 1 mm isotropic CT, the right from sagittal MRI with a resolution of 3.3 mm in the left–right direction. Top row: Subregion of the vertebra used for analysis. Middle row: Extreme points. Bottom row: Corpus edge and ligamentum flavum points.](../images/poi_preview.png)


| Symbol | Description |
|---|---|
| `POI` | Maps `(vertebra_id, subregion_id) → (x, y, z)` |
| `calc_centroids(seg_nii)` | Compute centroids for every label in a segmentation |
| `calc_poi_from_subreg_vert(vert, subreg)` | Compute POIs from paired vertebra + subregion segmentations |
| `POI.save(path)` | Serialise to JSON |
| `POI.load(path)` | Deserialise from JSON |
| `POI.to_global(ref)` | Convert from voxel to world (mm) coordinates |
| `POI.to_local(ref)` | Convert from world to voxel coordinates |
| `POI.save_mrk(ref)` | Saves the POI as Markup (to be used for 3D Slicer for example) |

Compute a simple poi object from a segmentation file
```python
from TPTBox import NII, calc_centroids

vert = NII.load("path/to/seg.nii.gz", True)
label_id = 20
second_stage = 1
# compute CMS
poi = calc_centroids(vert, second_stage=second_stage)
# The coordinate can be extracted by [Label-id, second_stage_id]
coords = poi[label_id, second_stage]
```

Compute a full set of anatomical landmarks. The registry of supported non-centroid POI strategies is exposed as
```python
from TPTBox.core.poi_fun.vertebra_pois_non_centroids import all_poi_functions

poi_full = calc_poi_from_subreg_vert(
    instance_nii,
    semantic_nii,
    subreg_id=list(all_poi_functions.keys()),
)
# export as a 3D Slicer markup file
poi_full.to_global().save_mrk("poi_as_markup.mrk.json", split_by_region=True, pointLabelsVisibility=True)
```


```python
from TPTBox import NII, POI, Location, POI_Global, calc_poi_from_subreg_vert
from TPTBox.core.vert_constants import v_name2idx
from TPTBox.segmentation.spineps import run_spineps_single

# This requires that spineps is installed
output_paths = run_spineps_single(
    "file-path-of_T2w.nii.gz",
    model_semantic="t2w",
    ignore_compatibility_issues=True,
)
out_spine = output_paths["out_spine"]
out_vert = output_paths["out_vert"]
semantic_nii = NII.load(out_spine, seg=True)
instance_nii = NII.load(out_vert, seg=True)

poi = calc_poi_from_subreg_vert(
    instance_nii,
    semantic_nii,
    subreg_id=[
        Location.Vertebra_Full,
        Location.Arcus_Vertebrae,
        Location.Spinosus_Process,
        Location.Costal_Process_Left,
        Location.Costal_Process_Right,
        Location.Superior_Articular_Left,
        Location.Superior_Articular_Right,
        Location.Inferior_Articular_Left,
        Location.Inferior_Articular_Right,
        # Location.Vertebra_Corpus_border, CT only
        Location.Vertebra_Corpus,
        Location.Vertebra_Disc,
        Location.Muscle_Inserts_Spinosus_Process,
        Location.Muscle_Inserts_Transverse_Process_Left,
        Location.Muscle_Inserts_Transverse_Process_Right,
        Location.Muscle_Inserts_Vertebral_Body_Left,
        Location.Muscle_Inserts_Vertebral_Body_Right,
        Location.Muscle_Inserts_Articulate_Process_Inferior_Left,
        Location.Muscle_Inserts_Articulate_Process_Inferior_Right,
        Location.Muscle_Inserts_Articulate_Process_Superior_Left,
        Location.Muscle_Inserts_Articulate_Process_Superior_Right,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,
        Location.Additional_Vertebral_Body_Middle_Superior_Median,
        Location.Additional_Vertebral_Body_Posterior_Central_Median,
        Location.Additional_Vertebral_Body_Middle_Inferior_Median,
        Location.Additional_Vertebral_Body_Anterior_Central_Median,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,
        Location.Additional_Vertebral_Body_Middle_Superior_Left,
        Location.Additional_Vertebral_Body_Posterior_Central_Left,
        Location.Additional_Vertebral_Body_Middle_Inferior_Left,
        Location.Additional_Vertebral_Body_Anterior_Central_Left,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,
        Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,
        Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,
        Location.Additional_Vertebral_Body_Middle_Superior_Right,
        Location.Additional_Vertebral_Body_Posterior_Central_Right,
        Location.Additional_Vertebral_Body_Middle_Inferior_Right,
        Location.Additional_Vertebral_Body_Anterior_Central_Right,
        Location.Ligament_Attachment_Point_Flava_Superior_Median,
        Location.Ligament_Attachment_Point_Flava_Inferior_Median,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    ],
)
poi = poi.round(2)
print("Vertebra T4 Vertebra Corpus Center of mass:", poi[v_name2idx["T4"], Location.Vertebra_Corpus])
print("The id number of T4 Vertebra_Corpus is ", v_name2idx["T4"], Location.Vertebra_Corpus.value)

# rescale/reorante local poi like nii
poi_new = poi.reorient(("P", "I", "R")).rescale((1, 1, 1))
# Local and global POIs can be rescaled to a target spacing with:
poi_new = poi.resample_from_to(other_nii_or_poi)

# local to global poi
global_poi = poi.to_global(itk_coords=True)
# You can save global pois in mrk.json format for import and editing in slicer.
global_poi.save_mrk("FILE.mrk.json", glyphScale=3.0)
# Import as a Markup in slicer; To make points editable you must click on the "lock" symbol under Markups - Control Points - Interaction

# Save in our format:
poi.save(poi_path)
# Loading local/global Poi
poi = POI.load(poi_path)
poi = POI_Global.load(poi_path)
```
