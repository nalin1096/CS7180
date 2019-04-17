# (generated with --quick)

import datetime
import image_preprocessing
import logging
from typing import Any, Dict, List, Tuple, Type

ImageDataPipeline: Type[image_preprocessing.ImageDataPipeline]
PIXEL_MAX: float
cv2: module
datetime: Type[datetime.datetime]
logger: logging.Logger
logging: module
np: module
os: module
tf: module

def compute_mae(Y_pred_i, Y_test_i) -> Any: ...
def compute_psnr(Y_pred_i, Y_test_i) -> Any: ...
def custom_evaluate_sony(test_dataflow, test_dataflow2, model, model_name, idp) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]: ...
def evaluate_model(test_dataflow, model, model_name) -> None: ...
def functional_sony() -> Dict[str, Any]: ...
def review_model(model, image_path: str) -> Any: ...
