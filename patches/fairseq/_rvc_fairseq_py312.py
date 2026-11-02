"""fairseq 0.12.2 在 Python 3.11+ 上的兼容补丁（由 _rvc_fairseq_py312.pth 自动加载）。

两个问题:
1. Python 3.11 起 dataclasses 把"可变默认值"的检查从
   ``isinstance(default, (list, dict, set))`` 收紧成
   ``default.__class__.__hash__ is None``，于是 fairseq 里
   ``field: XConfig = XConfig()`` 这种写法在 3.11/3.12 上直接抛
   ``ValueError: mutable default ... is not allowed: use default_factory``。
   处理：装饰前把不可哈希的默认值改写成 ``field(default_factory=...)``（deepcopy 同值对象）。

2. fairseq 用标准库 ``dataclasses.MISSING`` 表示必填字段，omegaconf 2.1 起只认
   自己的 ``omegaconf.MISSING``，否则抛
   ``ValidationError: Object of unsupported type: '_MISSING_TYPE'``。
   处理：仅对 fairseq 自己的 dataclass，把标准库 MISSING 换成 omegaconf.MISSING。

其他包的行为保持不变（只处理原本就会报错的两种写法）。
"""

import copy
import dataclasses

_original_dataclass = dataclasses.dataclass


def _is_stdlib_missing(value):
    return isinstance(value, type(dataclasses.MISSING))


def _patched_dataclass(cls=None, **kwargs):
    def wrap(c):
        is_fairseq_class = getattr(c, "__module__", "").startswith("fairseq")
        for name in list(getattr(c, "__annotations__", {})):
            if name not in c.__dict__:
                continue
            value = c.__dict__[name]
            if isinstance(value, dataclasses.Field):
                # 形如 `x: XConfig = field(default=XConfig())` 的写法：默认值藏在 Field 里
                if value.default_factory is dataclasses.MISSING and value.default is not dataclasses.MISSING:
                    try:
                        hash(value.default)
                    except TypeError:
                        value.default_factory = lambda _v=value.default: copy.deepcopy(_v)
                        value.default = dataclasses.MISSING
                continue
            # ClassVar / InitVar 不允许 default_factory，保持原样（例如 torch 内部 dataclass）
            annotation = c.__annotations__.get(name)
            text = annotation if isinstance(annotation, str) else str(annotation)
            if "ClassVar" in text or "InitVar" in text:
                continue
            if is_fairseq_class and _is_stdlib_missing(value):
                import omegaconf

                setattr(c, name, dataclasses.field(default=omegaconf.MISSING))
                continue
            try:
                hash(value)
            except TypeError:
                setattr(
                    c,
                    name,
                    dataclasses.field(default_factory=lambda _v=value: copy.deepcopy(_v)),
                )
        return _original_dataclass(c, **kwargs)

    if cls is None:
        return wrap
    return wrap(cls)


if dataclasses.dataclass is not _patched_dataclass:
    dataclasses.dataclass = _patched_dataclass
