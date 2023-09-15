# -*- coding: utf-8 -*-

from ctypes import cast, c_int, c_double, c_float, c_ulonglong, c_void_p

import numpy as np

from pyfr.backends.base import ComputeKernel, NotSuitableError
from pyfr.backends.openmp.provider import OpenMPKernelProvider
from pyfr.ctypesutil import LibWrapper


class XSMMWrappers(LibWrapper):
    _libname = 'xsmm'

    # Functions
    _functions = [
        (None, 'libxsmm_init'),
        (None, 'libxsmm_finalize'),
        (c_void_p, 'libxsmm_dfsspmdm_create', c_int, c_int, c_int, c_int,
         c_int, c_int, c_double, c_double, c_void_p, c_int, c_void_p),
        (c_void_p, 'libxsmm_sfsspmdm_create', c_int, c_int, c_int, c_int,
         c_int, c_int, c_float, c_float, c_void_p, c_int, c_void_p),
        (None, 'libxsmm_dfsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_sfsspmdm_execute', c_void_p, c_void_p, c_void_p),
        (None, 'libxsmm_dfsspmdm_destroy', c_void_p),
        (None, 'libxsmm_sfsspmdm_destroy', c_void_p),
        (c_ulonglong, 'libxsmm_timer_tick')
    ]


class OpenMPXSMMKernels(OpenMPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Kernel cache
        self._kerns = {}

        # Load and wrap libxsmm
        self._wrappers = XSMMWrappers()

        if backend.fpdtype == np.float64:
            self._nmod = 8
            self._createfn = self._wrappers.libxsmm_dfsspmdm_create
            self._execfn = self._wrappers.libxsmm_dfsspmdm_execute
            self._destroyfn = self._wrappers.libxsmm_dfsspmdm_destroy
        else:
            self._nmod = 16
            self._createfn = self._wrappers.libxsmm_sfsspmdm_create
            self._execfn = self._wrappers.libxsmm_sfsspmdm_execute
            self._destroyfn = self._wrappers.libxsmm_sfsspmdm_destroy

        # Init
        self._wrappers.libxsmm_init()

    def __del__(self):
        if hasattr(self, '_wrappers'):
            for blkptr in self._kerns.values():
                self._destroyfn(blkptr)

            self._wrappers.libxsmm_finalize()

    def mul(self, a, b, out, alpha=1.0, beta=0.0, nmex=None, a_facs=[]):

        # Check if a_facs present and if so it matches the original matrix
        if a_facs and a.nrow != a_facs[0].nrow and a.ncol != a_facs[-1].ncol:
            raise ValueError('Factor sizes do not match the original matrix')

        if len(a_facs) == 2:
            if np.allclose(a.get(), a_facs[0].get() @ a_facs[1].get()):
                print('decomposed matrix is close to the original', nmex)
            else:
                print('decomposed matrix is not equal to the original!!!!', nmex)
        if len(a_facs) == 3:
            if np.allclose(a.get(), a_facs[0].get() @ a_facs[1].get() @ a_facs[2].get()):
                print('decomposed matrix is close to the original', nmex)
            else:
                print('decomposed matrix is not equal to the original!!!!', nmex)
        if len(a_facs) == 4:
            if np.allclose(a.get(), a_facs[0].get() @ a_facs[1].get() @ a_facs[2].get() @ a_facs[3].get()):
                print('decomposed matrix is close to the original', nmex)
            else:
                print('decomposed matrix is not equal to the original!!!!', nmex)

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('libxsmm requires a constant a matrix')

        # Check n is suitable
        if b.leaddim % self._nmod != 0:
            raise NotSuitableError(f'libxsmm requires n % {self._nmod} = 0')

        # Check that beta is zero or one
        if beta != 0.0 and beta != 1.0:
            raise NotSuitableError('libxssm requires β = 0 or β = 1')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Cache key
        ckey = (a.mid, alpha, beta, b.nblocks, ldb, ldc)

        # Check the JIT kernel cache
        try:
            blkptr = self._kerns[ckey]
        except KeyError:
            c_is_nt = (beta == 0 and
                       out.nbytes >= 32*1024**2 and
                       self.backend.alignb >= 64) # pair C

            c_is_nt = 1 if nmex == 'disu' or nmex == 'tdivtpcorf' else 0 # pair B
            #c_is_nt = 0 # pair A

            print('xsmm kernel: ', nmex, 'non-temporal: ', c_is_nt)

            a_np = np.ascontiguousarray(a.get())
            m, k = a_np.shape

            timer_tick = cast(self._wrappers.libxsmm_timer_tick, c_void_p)

            # JIT and register an block leaddim size kernel for this matrix
            blkptr = self._createfn(m, b.leaddim, k, k, ldb, ldc, alpha,
                                    beta, a_np.ctypes.data, c_is_nt, timer_tick)
            #print('blkptr', blkptr)

            if a_facs:
                blkptr_facs = []
                for fac in a_facs:
                    mat = np.ascontiguousarray(fac.get())
                    m, kk = mat.shape
                    print('nmex shape', nmex, m, kk, ldb, ldc, alpha, beta)
                    if fac == a_facs[0]:
                        _beta = beta
                        _c_is_nt = c_is_nt
                    else:
                        _beta = 0
                        _c_is_nt = 0
                    blkptr_facs.append(
                        self._createfn(m, b.leaddim, kk, kk, ldb, ldc, alpha,
                                       _beta, mat.ctypes.data, _c_is_nt, timer_tick)
                    )
                    if not blkptr_facs[-1]:
                        raise NotSuitableError('libxssm unable to JIT a tensor product kernel')
            if not blkptr:
                raise NotSuitableError('libxssm unable to JIT a kernel')

            # Update the cache
            self._kerns[ckey] = blkptr

        # Obtain a pointer to the execute function
        execptr = cast(self._execfn, c_void_p).value

        if nmex == 'disu' and len(a_facs) == 4:
            krnl = 'disut'
        elif nmex == 'disu' and len(a_facs) == 3:
            krnl = 'disup'
        # Render our parallel wrapper kernel
        src = self.backend.lookup.get_template('batch-gemm').render(
            lib='xsmm', krnl=krnl if nmex=='disu' and a_facs else nmex
        )

        # Argument types for batch_gemm
        if nmex=='disu' and a_facs:
            argt = [np.intp]*len(a_facs) + [np.intp, np.int32]*3
        else:
            argt = [np.intp] + [np.intp, np.int32]*3

        # Build
        batch_gemm = self._build_kernel('batch_gemm', src, argt)

        name = 'par_xsmm_' + (nmex if nmex else '')
        print(name, 'xsmm.py')

        # Build
        #par_xsmm = self._build_kernel(name, src.replace('par_xsmm', name),
        #                              argt)

        # try executing xsmm kernels here
        #print('bnrow bleaddim', b.nrow, b.leaddim, out.nrow, out.leaddim)
        #print('a sizes', a.get().shape)
        #print('blkptr', blkptr)
        #bfake = np.ones((b.nrow, b.leaddim), dtype=np.float64)
        #outfake = np.zeros((out.nrow, out.leaddim), dtype=np.float64)

        #print('try to execute ', nmex)
        ##self._execfn(blkptr, bfake.ctypes.data, outfake.ctypes.data)
        #print('after execute ', nmex)

        class MulKernel(ComputeKernel):
            func_ptr = cast(batch_gemm, c_void_p).value
            e_ptr = cast(self._execfn, c_void_p).value
            b_ptr = blkptr
            if a_facs:
                bfac_ptr = blkptr_facs

            if nmex=='disu' and a_facs:
                print('nmex a_facs!!', nmex, *blkptr_facs, out.blocksz)
                def run(iself, queue):
                    batch_gemm(execptr, *blkptr_facs, b.nblocks, b, b.blocksz, out,
                               out.blocksz)
            else:
                def run(iself, queue):
                    batch_gemm(execptr, blkptr, b.nblocks, b, b.blocksz, out,
                               out.blocksz)

        return MulKernel()
