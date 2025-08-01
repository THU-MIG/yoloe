## Mapping between LVIS and COCO categories

The json file `coco_to_synset.json` provides a mapping from each COCO category
to a synset. The synset can then be used to find the corresponding category in
LVIS. Matching based on synsets (instead of category id) allows this mapping
to be correct even if LVIS category ids change (which will likely happen when
upgrading from LVIS release v0.5 to v1.0).
