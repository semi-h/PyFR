# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import (BaseAdvectionIntInters,
                                    BaseAdvectionMPIInters,
                                    BaseAdvectionBCInters)


class ACEulerIntInters(BaseAdvectionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.intcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class ACEulerSpIntInters(BaseAdvectionIntInters):
    def __init__(self, backend, lhs, rhs, elemap, cfg, nfpts, **kwargs):
        super().__init__(backend, lhs, rhs, elemap, cfg, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.spintcflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        print('spintinters nfpts')
        print(len(nfpts))
        print('lenperm', len(self._perm), type(self._perm), len(lhs))
        # gets the first element type, hex or tet
        elem = list(self.elemap.keys())[0]
        print('neles', self.elemap[elem].neles, elem)
        #print(self._perm)
        #print(lhs[self._perm])
        self.nfptsarr = self._be.matrix((len(nfpts), 1))
        self.nfptsarr.set(nfpts.reshape(-1, 1))
        #temparr = self.nfptsarr.get()
        #print(nfpts.reshape(-1, 1))
        #print(temparr)
        self.kernels['commpair_flux'] = lambda: self._be.kernel(
            'spintcflux', tplargs=tplargs, dims=[self.elemap[elem].neles],
            ul=self._scal_lhs, ur=self._scal_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            u=self.elemap[elem].scal_upts_inb, d=self.elemap[elem]._scal_fpts,
            nfpts=self.nfptsarr
        )


class ACEulerMPIInters(BaseAdvectionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.mpicflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class ACEulerBaseBCInters(BaseAdvectionBCInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.aceuler.kernels.bccflux')

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type)

        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            **self._external_vals
        )


class ACEulerInflowBCInters(ACEulerBaseBCInters):
    type = 'ac-in-fv'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c.update(self._exp_opts('uvw'[:self.ndims], lhs))


class ACEulerOutflowBCInters(ACEulerBaseBCInters):
    type = 'ac-out-fp'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c.update(self._exp_opts('p', lhs))


class ACEulerSlpWallBCInters(ACEulerBaseBCInters):
    type = 'slp-wall'


class ACEulerCharRiemInvBCInters(ACEulerBaseBCInters):
    type = 'ac-char-riem-inv'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c['niters'] = cfg.getint(cfgsect, 'niters', 4)
        self._tpl_c['bc-ac-zeta'] = cfg.getfloat(cfgsect, 'ac-zeta')
        tplc = self._exp_opts(
            ['p', 'u', 'v', 'w'][:self.ndims + 1], lhs
        )
        self._tpl_c.update(tplc)
