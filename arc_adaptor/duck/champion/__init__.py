"""冠军的三个视图模块，原样搬进来的部分。

来历（全部由 ``scripts/extract_champion.py`` 从 notebook cell 3 的补丁还原）：

===========================  ==========================================  ==========
模块                          来源                                         完整度
===========================  ==========================================  ==========
``animation.py``             ``inference/utils/animation.py``（新增文件）   完整 333 行
``frame_diff.py``            ``inference/utils/frame_diff.py``（新增文件）  完整 307 行
``lexicon.py``               ``cell 21`` 里 ``ARCAID3_LEXICON_SOURCE``      完整 31.9KB
``grid_utils.py``            **未公开**，按可验证语义补的两个符号           PORTER
===========================  ==========================================  ==========

没能搬进来的：``segmentation.py`` 只有 60 行 patch 片段，``segment_layer`` 的
主体不在里面。我们的 ``arc_adaptor/duck/views.py`` 顶替它，但缺
``rotation / rotational_symmetry / pose_hash / boundary`` 四个字段——其中
``boundary`` 是 ``frame_diff.py`` 判断"两个对象是否接触"的唯一依据，
不是可选装饰。

用法：这三个模块都不依赖冠军的运行时，只吃普通 list-of-list 网格。
"""

from . import animation, frame_diff, lexicon  # noqa: F401

__all__ = ["animation", "frame_diff", "lexicon"]
