# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements
from pyfr.shapes import _utoq, _utoq_pri


class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs
        solnsrc = self._soln_in_src_exprs

        # Source term kernel arguments
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        # Interpolation from elemental points
        if 'surf-flux' in self.antialias and self.basis.name == 'hex':
            M0facs = [self.basisq.opmat('M0')] + list(_utoq(self.basis.opmat('M7')))
            kernels['disu'] = lambda: self._be.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts, nmex='disu', a_facs=self.opmat_facs(M0facs)
            )
        elif 'surf-flux' in self.antialias and self.basis.name == 'pri':
            locp = self.basis.upts[::self.basis.nupts // (self.basis.order + 1)][:, 2]
            locq = self.basisq.upts[::self.basisq.nupts // (self.basisq.order + 1)][:, 2]
            print('locp, locq', locp, locq)
            M0facspri = [self.basisq.opmat('M0')] + list(_utoq_pri(locp, locq, self.basistri.opmat('M7')))
            kernels['disu'] = lambda: self._be.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts, nmex='disupri', a_facs=self.opmat_facs(M0facspri)
            )
        else:
            kernels['disu'] = lambda: self._be.kernel(
                'mul', self.opmat('M0'), self.scal_upts_inb,
                out=self._scal_fpts, nmex='disu'
            )

        if fluxaa and self.basis.name == 'hex':
            kernels['qptsu'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_inb,
                out=self._scal_qpts, nmex='qptsu',
                a_facs=self.opmat_facs(_utoq(self.basis.opmat('M7')))
            )
        elif fluxaa and self.basis.name == 'pri':
            locp = self.basis.upts[::self.basis.nupts // (self.basis.order + 1)][:, 2]
            locq = self.basisq.upts[::self.basisq.nupts // (self.basisq.order + 1)][:, 2]
            kernels['qptsu'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_inb,
                out=self._scal_qpts, nmex='qptsu',
                a_facs=self.opmat_facs(_utoq_pri(locp, locq, self.basistri.opmat('M7')))
            )
        elif fluxaa:
            kernels['qptsu'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_inb,
                out=self._scal_qpts, nmex='qptsu',
            )

        # First flux correction kernel
        if fluxaa and self.basis.name == 'hex':
            M9facs = [np.kron(np.eye(self.ndims), m)
                      for m in _utoq(self.basis.opmat('M8'))]
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
                out=self.scal_upts_outb, nmex='tdivtpcorf',
                a_facs=self.opmat_facs([self.basis.opmat('M1 - M3*M2')] + M9facs)
            )
        elif fluxaa and self.basis.name == 'pri':
            locp = self.basis.upts[::self.basis.nupts // (self.basis.order + 1)][:, 2]
            locq = self.basisq.upts[::self.basisq.nupts // (self.basisq.order + 1)][:, 2]
            M9facspri = [np.kron(np.eye(self.ndims), m)
                      for m in _utoq_pri(locq, locp, self.basistri.opmat('M8'))]
            print('tdivtpcorf  shape', self.basis.opmat('M1-M3*M2').shape)
            print('tdivtpcorf basistri', self.basistri.opmat('M8').shape, self.basistri.order)
            print('utoqprishapes', [m.shape for m in _utoq_pri(locq, locp, self.basistri.opmat('M8'))])
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
                out=self.scal_upts_outb, nmex='tdivtpcorf',
                a_facs=self.opmat_facs([self.basis.opmat('M1 - M3*M2')] + M9facspri)
            )
        elif fluxaa:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M9'), self._vect_qpts,
                out=self.scal_upts_outb, nmex='tdivtpcorf'
            )
        else:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb, nmex='tdivtpcorf'
            )

        # Second flux correction kernel
        if 'surf-flux' in self.antialias and self.basis.name == 'hex':
            M3facs = list(_utoq(self.basis.opmat('M8'))) + [self.basisq.opmat('M3')]
            kernels['tdivtconf'] = lambda: self._be.kernel(
                'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
                beta=1.0, nmex='tdivtconf', a_facs=self.opmat_facs(M3facs)
            )
        elif 'surf-flux' in self.antialias and self.basis.name == 'pri':
            locp = self.basis.upts[::self.basis.nupts // (self.basis.order + 1)][:, 2]
            locq = self.basisq.upts[::self.basisq.nupts // (self.basisq.order + 1)][:, 2]
            M3facspri = list(_utoq_pri(locq, locp, self.basistri.opmat('M8'))) + [self.basisq.opmat('M3')]
            kernels['tdivtconf'] = lambda: self._be.kernel(
                'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
                beta=1.0, nmex='tdivtconf', a_facs=self.opmat_facs(M3facspri)
            )
        else:
            kernels['tdivtconf'] = lambda: self._be.kernel(
                'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
                beta=1.0, nmex='tdivtconf'
            )

        # Transformed to physical divergence kernel + source term
        plocupts = self.ploc_at('upts') if plocsrc else None
        solnupts = self._scal_upts_cpy if solnsrc else None

        if solnsrc:
            kernels['copy_soln'] = lambda: self._be.kernel(
                'copy', self._scal_upts_cpy, self.scal_upts_inb
            )

        kernels['negdivconf'] = lambda: self._be.kernel(
            'negdivconf', tplargs=srctplargs,
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
            rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts, u=solnupts,
            d=self._scal_fpts
        )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln():
                mul = self._be.kernel(
                    'mul', self.opmat('M10'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            kernels['filter_soln'] = filter_soln
