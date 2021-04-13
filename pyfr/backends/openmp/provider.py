# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.openmp.compiler import SourceModule
from pyfr.backends.openmp.generator import OpenMPKernelGenerator
from pyfr.util import memoize


class OpenMPKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        print('name', name)
        if name=='spintcflux':
            f = open('spintcflux.c', 'w')
            f.write(src)
            f.close()
        if name=='tfluxlin':
            f = open('tfluxlin.c', 'w')
            f.write(src)
            f.close()
        if name=='tflux':
            f = open('tflux.c', 'w')
            f.write(src)
            f.close()
        mod = SourceModule(src, self.backend.cfg)
        return mod.function(name, restype, argtypes)


class OpenMPPointwiseKernelProvider(OpenMPKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenMPKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            if any(isinstance(arg, str) for arg in arglst):
                def run(self, queue, **kwargs):
                    fun(*[kwargs.get(ka, ka) for ka in arglst])
            else:
                def run(self, queue, **kwargs):
                    fun(*arglst)

        return PointwiseKernel()
