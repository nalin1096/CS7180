# (generated with --quick)

import datetime
import logging
from typing import Any, Dict, NoReturn, Optional, Tuple, Type, TypeVar

Adam: module
COVM: Any
ImageDataGenerator: module
MEANM: Any
ModelCheckpoint: module
cifar10: module
datetime: Type[datetime.datetime]
fcov: str
fmean: str
logger: logging.Logger
logging: module
np: module
os: module
pickle: module
tf: module

AnyStr = TypeVar('AnyStr', str, bytes)
_T1 = TypeVar('_T1')

def bl(image, sample = ...) -> Any: ...
def bl_cd(image, sample = ...) -> Any: ...
def bl_cd_pn(image, sample = ...) -> Any: ...
def bl_cd_pn_ag(image, sample = ...) -> Any: ...
def callbacks(model_type) -> list: ...
def create_patch(X, ps = ...) -> Any: ...
def enable_cloud_log(level = ...) -> None: ...
def fit_model(dataflow, model: _T1, imgtup, model_id, lr = ..., epochs = ...) -> Tuple[_T1, Any]: ...
def fit_model_ngpus(X_train, Y_train, model, imgtup, lr = ..., epochs = ...) -> None: ...
def main() -> None: ...
def main_ngpus() -> None: ...
def mean_absolute_error(y_true, y_pred) -> Any: ...
def model02() -> Dict[str, Any]: ...
def model_predict(model, X_test, imgtup) -> Any: ...
def plot_images(name, X_test, Y_pred, Y_true) -> None: ...
def plot_loss(fpath, history) -> None: ...
def read_pickle(fpath) -> Any: ...
def review_model(X_test, Y_true, model, history, imgtup, num_images = ...) -> None: ...
def review_sony_model(results, imgnum) -> None: ...
def run_simulation(fcov, fmean) -> None: ...
def run_sony_images(model, model_type) -> None: ...
def simple_sony() -> Any: ...
def urljoin(base: AnyStr, url: Optional[AnyStr], allow_fragments: bool = ...) -> AnyStr: ...
def valid_sample() -> NoReturn: ...