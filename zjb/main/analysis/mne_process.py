import mne
import numpy as np
from mne import Forward, SourceSpaces
from mne.bem import ConductorModel
from mne.io import Raw, RawArray
from traits.api import Bool, HasPrivateTraits, HasRequiredTraits, Str, Union

from zjb._traits.types import Instance
from .base import AnalyzerBase
from ..data.regionmapping import SurfaceRegionMapping
from ..data.series import MNEsimSeries
from ..data.series import RegionalTimeSeries


class CreateMNESignals(HasPrivateTraits, HasRequiredTraits):
    """
    从仿真的脑区神经活动创建MNE格式的电生理信号

    Attributes
    ----------
    fwd : Forward
        mne正向解实例
    raw : Raw
        mne的传感器空间数据实例，包含mne的采集信息info
    mapping : SurfaceRegionMapping
        皮层-脑区映射实例
    regional_ts : RegionalTimeSeries
        脑区时间序列实例
    src : path-like | SourceSpaces实例
        可以是源空间文件的路径，或是SourceSpaces实例
    bem : path-like | ConductorModel实例
        边界元模型，表明头模型每一层的导电率
    trans : str | None
        头模型 head<->MRI 转换的文件.
    eeg : Bool
        表明是否输出eeg的数据
    meg : Bool
        表明是否输出meg的数据
    """

    fwd = Union(Str, Instance(Forward))

    raw = Union(Str, Instance(Raw), Instance(RawArray))

    mapping = Instance(SurfaceRegionMapping, required=True)

    regional_ts = Instance(RegionalTimeSeries, required=True)

    src = Union(Str, Instance(SourceSpaces))

    bem = Union(Str, Instance(ConductorModel))

    trans = Str("fsaverage")

    eeg = Bool(True)

    meg = Bool(True)

    def _default_raw(self):
        """如果没有raw，则生成默认的raw"""
        default_channels = 32
        sampling_freq = 200  # in Hertz
        info = mne.create_info(default_channels, sfreq=sampling_freq)
        times = np.linspace(0, 1, sampling_freq, endpoint=False)
        sine = np.sin(20 * np.pi * times)
        cosine = np.cos(10 * np.pi * times)
        data = np.array([sine, cosine])

        info = mne.create_info(
            ch_names=["10 Hz sine", "5 Hz cosine"],
            ch_types=["misc"] * 2,
            sfreq=sampling_freq,
        )

        self.raw = mne.io.RawArray(data, info)

    def _build_fwd(self):
        """如果没有fwd，则通过src, trans和bem生成一个fwd"""
        if not self.bem or not self.src:
            raise ValueError(
                """When you generate fwd, you must have src, trans, and bem"""
            )

        self.fwd = mne.make_forward_solution(
            self.raw.info,
            trans=self.trans,
            src=self.src,
            bem=self.bem,
            meg=self.meg,
            eeg=self.eeg,
            mindist=5.0,
            verbose=True,
        )

    def __call__(self):
        dt, unit = AnalyzerBase.extract_dt_unit(self.regional_ts)

        if not self.raw:
            self._default_raw()
        elif isinstance(self.raw, str):
            self.raw = mne.io.read_raw_fif(self.raw)

        self.raw.resample(sfreq=1 / dt)

        if not self.fwd:
            self._build_fwd()
        elif isinstance(self.fwd, str):
            self.fwd = mne.read_forward_solution(self.fwd)

        vertices_ts_data = self.regional_ts.data.T[self.mapping.data - 1]

        vertices_L, vertices_R = np.split(
            np.array(range(self.mapping.data.shape[0])), 2
        )
        vertices = [vertices_L] + [vertices_R]

        stc = mne.SourceEstimate(
            data=vertices_ts_data,
            tmin=self.regional_ts.start_time,
            tstep=dt,
            vertices=vertices,
        )

        raw_sim = mne.simulation.simulate_raw(
            self.raw.info, stc, forward=self.fwd, verbose=True
        )
        mne_sim_series = MNEsimSeries()
        mne_sim_series.rawarray = raw_sim

        return mne_sim_series
