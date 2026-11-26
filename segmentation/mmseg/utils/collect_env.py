# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    if sys.platform == 'win32':
        env_info = _collect_windows_env()
    else:
        env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


def _collect_windows_env():
    from distutils.errors import DistutilsPlatformError

    try:
        return collect_base_env()
    except (DistutilsPlatformError, UnicodeDecodeError):
        # MMCV 1.x probes MSVC even when the model uses prebuilt wheels.
        import mmcv
        import torch

        env_info = {
            'sys.platform': sys.platform,
            'Python': sys.version.replace('\n', ''),
            'CUDA available': torch.cuda.is_available(),
            'PyTorch': torch.__version__,
            'CUDA runtime': torch.version.cuda,
            'MMCV': mmcv.__version__,
            'MSVC': 'Compiler information unavailable',
        }
        for index in range(torch.cuda.device_count()):
            env_info[f'GPU {index}'] = torch.cuda.get_device_name(index)
        return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
