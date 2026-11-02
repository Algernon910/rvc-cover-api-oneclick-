# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from dataclasses import MISSING as _DC_MISSING

from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _field_default(field):
    """Python 3.11+ 兼容：取值时同时考虑 default / default_factory。

    原实现直接读 ``field.default``，而本环境为绕开 3.11 的可变默认值检查，
    已把 ``x: XConfig = XConfig()`` 改写成 ``field(default_factory=...)``，
    此时 ``field.default`` 变回标准库 MISSING。新版 hydra/omegaconf 不再接受
    ``_MISSING_TYPE`` 对象，会直接抛
    ``ValidationError: Object of unsupported type: '_MISSING_TYPE'``。
    """
    if field.default is not _DC_MISSING:
        return field.default, True
    if field.default_factory is not _DC_MISSING:
        return field.default_factory(), True
    return None, False


def hydra_init(cfg_name="config") -> None:

    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)

    for k, fld in FairseqConfig.__dataclass_fields__.items():
        v, has_default = _field_default(fld)
        if not has_default:
            # 没有默认值的字段(必填项)无需注册，跳过即可
            continue
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise


def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about"""

    from fairseq.registry import REGISTRIES
    from fairseq.tasks import TASK_DATACLASS_REGISTRY
    from fairseq.models import ARCH_MODEL_NAME_REGISTRY, MODEL_DATACLASS_REGISTRY
    from fairseq.dataclass.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in FairseqConfig.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)

            if k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            elif k == "model":
                name = ARCH_MODEL_NAME_REGISTRY.get(name, name)
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k in REGISTRIES:
                dc = REGISTRIES[k]["dataclass_registry"].get(name)

            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)
