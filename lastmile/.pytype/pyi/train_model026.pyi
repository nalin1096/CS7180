# (generated with --quick)

import datetime
import image_preprocessing
import logging
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

Adam: module
ImageDataPipeline: Type[image_preprocessing.ImageDataPipeline]
ModelCheckpoint: module
RaiseDataGenerator: Type[image_preprocessing.RaiseDataGenerator]
cv2: module
datetime: Type[datetime.datetime]
json: module
logger: logging.Logger
logging: module
np: module
os: module
tf: module

AnyStr = TypeVar('AnyStr', str, bytes)

def callbacks(model_type) -> list: ...
def enable_cloud_log(level = ...) -> None: ...
def fit_model(train_dataflow, val_dataflow, mod, imgproc, lr, epochs) -> Tuple[Any, Any]: ...
def functionial_sony() -> Dict[str, Any]: ...
def main() -> None: ...
def mean_absolute_error(y_true, y_pred) -> Any: ...
def run_simulation(mod: dict) -> None: ...
def urljoin(base: AnyStr, url: Optional[AnyStr], allow_fragments: bool = ...) -> AnyStr: ...
