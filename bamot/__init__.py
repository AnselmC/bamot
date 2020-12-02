import os
import sys
from pathlib import Path

MODULE_PATH = Path(os.path.dirname(__file__)).absolute()
SUPERPOINT_PATH: Path = MODULE_PATH / "thirdparty" / "SuperPoint"
SUPERPOINT_SETTINGS: Path = SUPERPOINT_PATH / "superpoint" / "settings.py"
# ./setup.sh script in thirdparty `SuperPoint` repo creates `settings.py`.
# The file won't be there if the script wasn't run and is not necessary
sys.path.append(SUPERPOINT_PATH.as_posix())
POINTNET_BASE_PATH: Path = MODULE_PATH / "thirdparty" / "pointnet2"
POINTNET_MAIN_PATH: Path = POINTNET_BASE_PATH / "pointnet2"
POINTNET_OPS_PATH: Path = POINTNET_BASE_PATH / "pointnet2_ops"
sys.path.append(POINTNET_BASE_PATH.as_posix())
sys.path.append(POINTNET_MAIN_PATH.as_posix())
sys.path.append(POINTNET_OPS_PATH.as_posix())
try:
    import superpoint.settings
except ModuleNotFoundError:
    with open(SUPERPOINT_SETTINGS, "w") as f:
        f.write('EXPER_PATH=""')
