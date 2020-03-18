# -*- coding: utf-8 -*-

from gimmik import generate_mm
from ctypes import cast, c_void_p
import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider


class OpenMPGiMMiKKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.max_nnz = backend.cfg.getint('backend-openmp', 'gimmik-max-nnz',
                                          512)

    def mul(self, a, b, out, alpha=1.0, beta=0.0, nmex=None):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Check that A is reasonably sparse
        if np.count_nonzero(a.get()) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Generate the GiMMiK kernel
        src = generate_mm(a.get(), dtype=a.dtype, platform='c-omp',
                          alpha=alpha, beta=beta)

        name = 'gimmik_mm_' + (nmex if nmex else '')

        gimmik_mm = self._build_kernel(name, src.replace('gimmik_mm', name),
                                       [np.int32] + [np.intp, np.int32]*2)
        print(name, gimmik_mm)

        class MulKernel(ComputeKernel):
            func_ptr = cast(gimmik_mm, c_void_p).value

            def run(self, queue):
                gimmik_mm(b.ncol, b, b.leaddim, out, out.leaddim)

        return MulKernel()
