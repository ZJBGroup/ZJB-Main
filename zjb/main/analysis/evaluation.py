from enum import Enum

import numpy as np
from scipy.signal import butter, correlate, detrend, filtfilt, hilbert, resample
from sklearn.manifold import SpectralEmbedding
from traits.api import Array, Bool, Dict, Enum, Float, HasTraits, Instance, Int, Tuple

from zjb.main.analysis.base import AnalyzerBase
from zjb.main.data.series import TimeSeries


def pearson_correlation(timeseries: TimeSeries):
    """计算节点之间的皮尔森相关系数"""
    corr_result = np.corrcoef(timeseries.data, rowvar=False)
    return corr_result


def temporal_covariance(timeseries: TimeSeries):
    """节点间的协方差"""
    covar_result = np.cov(timeseries.data, rowvar=False)
    return covar_result


def kld(x: np.ndarray, y: np.ndarray):
    kld_result = 0.5 * (np.sum(x * np.log(x / y)) + np.sum(y * np.log(y / x)))
    return kld_result


class FCDAnalysis(HasTraits):
    """计算节点的动态功能连接"""

    ts_emp: TimeSeries = Instance(TimeSeries, input=True)

    # FC analysis methods (PCC, CC, MTD)
    method = Enum("PCC", "CC", "MTD", input=True)

    # using sliding window analysis
    swc = Bool(True, input=True)

    # Sliding window length (s)
    sw = Float(1.0, input=True)

    # Spanning between two consecutive sliding window (s)
    sp = Float(0.5, input=True)

    f_lo = Float(0.01, input=True)

    f_hi = Float(1, input=True)

    __return__ = Array(dtype=np.float32, output=True)

    def __call__(self):
        if not self.ts_emp:
            raise ValueError("Lack of ts input")
        if (
            self.swc
            and (self.ts_emp.data.shape[0] - 1) * self.ts_emp.sample_period < self.sw
        ):
            raise ValueError("Window too large")

        dt, unit = AnalyzerBase.extract_dt_unit(self.ts_emp)
        fs, tr = (
            1 / dt,
            (self.ts_emp.data.shape[0] - 1) * self.ts_emp.sample_period * unit,
        )
        data = self._apply_bandpass_filter(self.ts_emp.data, fs)
        dFC = self._compute_dFC_series(data, tr, fs)
        return dFC

    def _compute_dFC_series(self, data: np.ndarray, tr: np.ndarray, fs: float):
        met_fc = {
            "PCC": self.pearson_corr_coeff,
            "CC": self.coherence_connectivty,
            "MTD": self.mul_temporal_derivatives,
        }
        if not self.swc or np.isnan(fs):
            self.sw, self.sp = 0, 0
            data = data.reshape((data.shape[0], 1, data.shape[1]))
            dFC = np.array([met_fc[self.method](i) for i in data])
        else:
            windows = self.__slide_temporal_windows(data, tr, fs)
            dFC = np.array([met_fc[self.method](i) for i in windows])

        return dFC

    def _apply_bandpass_filter(
        self, x: np.ndarray, fs: float, order: int = 2
    ) -> np.ndarray:
        if np.isnan(fs):
            return x
        b, a = butter(order, [self.f_lo, self.f_hi], btype="band", fs=fs)
        x_filted = filtfilt(b, a, x, axis=0)
        return x_filted

    def __slide_temporal_windows(
        self, x: np.ndarray, t: float, fs: float, axis: int = 0
    ) -> np.ndarray:
        x = np.swapaxes(x, 0, axis)
        sw_size = int(self.sw * fs)
        sp_size = int(self.sp * fs)
        n_w = int((t - self.sw) / self.sp) + 1
        w_shape = (n_w, sw_size, *x.shape[1:])
        w_stride = (sp_size * x.strides[0], x.strides[0], *x.strides[1:])
        windows = np.lib.stride_tricks.as_strided(x, shape=w_shape, strides=w_stride)
        return windows

    @staticmethod
    def pearson_corr_coeff(x: np.ndarray, axis_1: bool = True) -> np.ndarray:
        fcm = np.corrcoef(x, rowvar=int(not axis_1))
        return fcm

    @staticmethod
    def cosine_similarity(x: np.ndarray, axis_1: bool = False) -> np.ndarray:
        norm = np.linalg.norm(x, axis=int(not axis_1))
        dot = np.matmul(x, x.T)
        simularity = dot / np.outer(norm, norm)
        return simularity

    @staticmethod
    def coherence_connectivty(x: np.ndarray) -> np.ndarray:
        phi = np.angle(hilbert(x))
        d_phi = np.abs(phi[:, :, np.newaxis] - phi[:, np.newaxis, :])
        d_phi = np.mean(d_phi, axis=0)
        fcm = np.cos(d_phi)
        return fcm

    @staticmethod
    def mul_temporal_derivatives(x: np.ndarray) -> np.ndarray:
        dt = x[1:, :] - x[:-1, :]
        std = np.std(x, axis=0)
        dt_prod = np.multiply(dt[:, :, np.newaxis], dt[:, np.newaxis, :])
        std_prod = np.outer(std, std)
        fcm = np.mean(dt_prod / std_prod, axis=0)
        return fcm

    @staticmethod
    def eigval_eigvec_extraction(x: np.ndarray, n: int = 1) -> tuple:
        eigval_m, eigvect_m = np.linalg.eigh(x)
        eigvals = np.zeros(n, dtype=np.float32)
        eigvects = np.zeros((n, eigvect_m.shape[0]), dtype=np.float32)

        for ei in range(n):
            max_i = np.argmax(eigval_m)
            eigvals[ei] = eigval_m[max_i]
            eigvects[ei] = np.abs(eigvect_m[:, max_i])
            eigval_m[max_i] = 0

        return eigvals, eigvects


class FCDMatrix(FCDAnalysis):
    """计算节点的动态功能连接矩阵和稳定周期指数"""

    comparison = Enum("CS-LEi", "CS-UT", "PCC-LEi", "PCC-UT", input=True)

    n_eig = Int(3, input=True)

    __return__ = Tuple(
        Array(dtype=np.float32),  # FCD matrix
        Dict(),  # Eigenvalues of stable periods
        Dict(),  # Eigenvectors of stable periods
    )

    def __call__(self):
        if not self.ts_emp:
            raise ValueError("Lack of ts input")
        if (
            self.swc
            and (self.ts_emp.data.shape[0] - 1) * self.ts_emp.sample_period < self.sw
        ):
            raise ValueError("Window too large")

        dt, unit = AnalyzerBase.extract_dt_unit(self.ts_emp)
        fs, tr = (
            1 / dt,
            (self.ts_emp.data.shape[0] - 1) * self.ts_emp.sample_period * unit,
        )
        data = self._apply_bandpass_filter(self.ts_emp.data, fs)
        dFC = self._compute_dFC_series(data, tr, fs)
        fcd_matrix = self._compute_FCD_matrix(dFC)
        eigvals, eigvects = self._extract_stable_eigs(fcd_matrix, fs)
        return fcd_matrix, eigvals, eigvects

    def _compute_FCD_matrix(self, dFC: np.ndarray):
        comparative_methods = {
            "CS": self.cosine_similarity,
            "PCC": self.pearson_corr_coeff,
        }
        extraction_methods = {
            "LEi": self.__leading_eigvectors,
            "UT": self.__upper_triangles,
        }
        met_c = comparative_methods[self.comparison.split("-")[0]]
        met_e = extraction_methods[self.comparison.split("-")[1]]
        fcd_matrix = met_c(met_e(dFC), axis_1=False)
        return fcd_matrix

    def _extract_stable_eigs(self, fcd_matrix: np.ndarray, fs: float) -> tuple:
        stable_eigval = {}
        stable_eigvect = {}
        xir, xir_cutoff = self.__spectral_embedding(fcd_matrix)
        periods, periods_t = self.__stable_periods_identification(fs, xir, xir_cutoff)

        for i in range(periods.shape[0]):
            name = "-".join(periods_t[i].astype(str))
            data = self.ts_emp.data[periods[i][0] : periods[i][1], :]
            fcd_matrix_partial = np.corrcoef(data, rowvar=0)
            val, vect = self.eigval_eigvec_extraction(fcd_matrix_partial, self.n_eig)
            stable_eigval[name] = val
            stable_eigvect[name] = vect

        return stable_eigval, stable_eigvect

    def __leading_eigvectors(self, dFC: np.ndarray) -> np.ndarray:
        V_s = np.ones(dFC.shape[:2], dtype=np.float32)
        for i in range(dFC.shape[0]):
            _, V_1 = self.eigval_eigvec_extraction(dFC[i], n=1)
            V_s[i] = V_1[0]
        return V_s

    def __upper_triangles(self, dFC: np.ndarray) -> np.ndarray:
        V_s = np.ones((dFC.shape[0], dFC.shape[1] ** 2), dtype=np.float32)
        for i in range(dFC.shape[0]):
            V_1 = np.triu(dFC[i]).flatten()
            V_s[i] = V_1
        return V_s

    def __spectral_embedding(self, fcd_matrix: np.ndarray, n_dim: int = 2) -> tuple:
        se = SpectralEmbedding(n_dim, affinity="precomputed")
        xi = se.fit_transform(fcd_matrix - fcd_matrix.min())
        xir = AnalyzerBase.compute_norms(xi, True, 1)
        xir_cutoff = 0.5 * np.sort(xir)[-1]
        return xir, xir_cutoff

    def __stable_periods_identification(
        self, fs: float, xir: np.ndarray, xir_cutoff: np.ndarray
    ) -> dict:
        sw_size = int(self.sw * fs)
        sp_size = int(self.sp * fs)

        if not np.where(xir < xir_cutoff)[0].size:
            periods = np.zeros((1, 2), dtype=np.int32)
            periods[0, 0] = 1 * sw_size
            periods[0, 1] = (xir.shape[0] - 1) * sp_size + sw_size
        else:
            jp = np.where(np.diff(xir > xir_cutoff).astype(int) == 1)[0] + 1
            periods = np.zeros((jp.shape[0] // 2, 2), dtype=np.int32)
            for i in range(periods.shape[0]):
                periods[i, 0] = jp[i * 2] * sp_size
                periods[i, 1] = jp[i * 2 + 1] * sp_size + sw_size
        periods_t = np.round(periods * 1 / fs, 3)
        return periods, periods_t
