from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .t2m_36dim import TM2TMetrics36Dim
from .mm import MMMetrics
from .mm_36dim import MMMetrics36Dim
from .m2t import M2TMetrics
from .m2m import PredMetrics


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints

        data_name = datamodule.name
        if data_name in ["humanml3d", "kit"]:
            self.TM2TMetrics = TM2TMetrics(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
            self.M2TMetrics = M2TMetrics(
                cfg=cfg,
                w_vectorizer=datamodule.hparams.w_vectorizer,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
            self.MMMetrics = MMMetrics(
                cfg=cfg,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        elif data_name in ("customdata_long", "customdata_all",
                           "customdata_combined"):
            # Use 36-dim evaluator for custom motion data
            self.TM2TMetrics = TM2TMetrics36Dim(
                cfg=cfg,
                dataname=data_name,
                diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
            # Skip M2TMetrics for custom data (uses hardcoded "t2m" path)
            self.MMMetrics = MMMetrics36Dim(
                cfg=cfg,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )

        self.MRMetrics = MRMetrics(
            njoints=njoints,
            jointstype=cfg.DATASET.JOINT_TYPE,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
        )
        self.PredMetrics = PredMetrics(
            cfg=cfg,
            njoints=njoints,
            jointstype=cfg.DATASET.JOINT_TYPE,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            task=cfg.model.params.task,
        )
