import logging
from lvis import LVIS, LVISResults, LVISEval

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "./data/lvis_val_100.json"
RESULT_PATH = "./data/lvis_results_100.json"

ANN_TYPE = 'bbox'

lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH, ANN_TYPE)
lvis_eval.run()
lvis_eval.print_results()
