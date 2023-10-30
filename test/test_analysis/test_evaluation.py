import numpy as np
from traits.api import Enum

from zjb.main.analysis.evaluation import FCDAnalysis
from zjb.main.data.series import RegionalTimeSeries, TimeUnit


def _init_regional_time_series():
    ts_data = np.load(".\Tmax1000-wen0.0000-weg0.0000-00.npy").T
    return RegionalTimeSeries(
        data=ts_data,
        sample_unit=TimeUnit.SECOND,
        sample_period=2,
    )


def test_FCDAnalysis():
    "测试FCD是否正常执行"

    ts = _init_regional_time_series()
    fcd_analysis = FCDAnalysis(ts_emp=ts, method="CC", sw=3, f_hi=0.1)
    dFC = fcd_analysis()
