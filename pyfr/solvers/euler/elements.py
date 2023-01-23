# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements(object):
    formulations = ['std', 'dual']

    privarmap = {2: ['rho', 'u', 'v', 'p'],
                 3: ['rho', 'u', 'v', 'w', 'p']}

    convarmap = {2: ['rho', 'rhou', 'rhov', 'E'],
                 3: ['rho', 'rhou', 'rhov', 'rhow', 'E']}

    dualcoeffs = convarmap

    visvarmap = {
        2: [('density', ['rho']),
            ('velocity', ['u', 'v']),
            ('pressure', ['p'])],
        3: [('density', ['rho']),
            ('velocity', ['u', 'v', 'w']),
            ('pressure', ['p'])]
    }

    @staticmethod
    def pri_to_con(pris, cfg):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho] + rhovs + [E]

    @staticmethod
    def con_to_pri(cons, cfg):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tflux')
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tfluxlin')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Set external arguments and values specific to the flux kernels
        extrnl_args = self._external_args.copy()
        extrnl_vals = self._external_vals.copy()
        extrnl_args['disu_exec'] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['tdivtpcorf_exec'] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['tdivtconf_exec'] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['disu_blkk'] = 'in fptr vptr'
        extrnl_args['tdivtconf_blkk'] = 'in fptr vptr'
        extrnl_args['tdivtpcorf_blkk'] = 'in fptr vptr'

        # Static code injection out of main loop
        inject = list()

        # Common arguments
        if 'flux' in self.antialias:
            uu = lambda s: self._slice_mat(self.scal_upts_inb, s)
            u = lambda s: self._slice_mat(self._scal_qpts, s)
            uout = lambda s: self._slice_mat(self.scal_upts_outb, s)
            d = lambda s: self._slice_mat(self._scal_fpts, s)
            f = lambda s: self._slice_mat(self._vect_qpts, s)
            pts, npts = 'qpts', self.nqpts

            extrnl_args['qptsu_exec'] = 'in fptr libxsmm_xfsspmdm_execute'
            extrnl_args['qptsu_blkk'] = 'in fptr vptr'
            for i in range(self.ndims):
                extrnl_args['qptsu_blkk'+str(i)] = 'in fptr vptr'
            for i in range(self.ndims+1):
                extrnl_args['tdivtpcorf_blkk'+str(i)] = 'in fptr vptr'
                extrnl_args['tdivtconf_blkk'+str(i)] = 'in fptr vptr'
            extrnl_args['uu'] = f'in fpdtype_t[{self.nvars}]'
            #extrnl_vals['uu'] = self._slice_mat(self.scal_upts_inb, s)
            inject.append('''
                fpdtype_t buffqpts[nqpts*BLK_SZ*nvars];
                qptsu_exec(qptsu_blkk2, uu_v+ib*BLK_SZ*nvars*nupts, u_v+ib*BLK_SZ*nvars*nqpts);
                qptsu_exec(qptsu_blkk1, u_v+ib*BLK_SZ*nvars*nqpts, buffqpts);
                qptsu_exec(qptsu_blkk0, buffqpts, u_v+ib*BLK_SZ*nvars*nqpts);
                //qptsu_exec(qptsu_blkk, uu_v+ib*BLK_SZ*nvars*nupts, u_v+ib*BLK_SZ*nvars*nqpts);
                fpdtype_t buffarr[nqpts*ndims*BLK_SZ*nvars];
                ''')
            inject.append('''
                //tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                fpdtype_t buffuout[nqpts*ndims*BLK_SZ*nvars];
                fpdtype_t buffuout2[nqpts*ndims*BLK_SZ*nvars];
                tdivtpcorf_exec(tdivtpcorf_blkk3, buffarr, buffuout);
                tdivtpcorf_exec(tdivtpcorf_blkk2, buffuout, buffuout2);
                tdivtpcorf_exec(tdivtpcorf_blkk1, buffuout2, buffuout);
                tdivtpcorf_exec(tdivtpcorf_blkk0, buffuout, uout_v+ib*BLK_SZ*nvars*nupts);

                //tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);
                //fpdtype_t buffqout[nqpts*BLK_SZ*nvars];
                //fpdtype_t buffqout2[nqpts*BLK_SZ*nvars];
                tdivtconf_exec(tdivtconf_blkk3, d_v+ib*BLK_SZ*nvars*nfpts, buffuout);
                tdivtconf_exec(tdivtconf_blkk2, buffuout, buffuout2);
                tdivtconf_exec(tdivtconf_blkk1, buffuout2, buffuout);
                tdivtconf_exec(tdivtconf_blkk0, buffuout, uout_v+ib*BLK_SZ*nvars*nupts);
                _ny = nupts;
                ''')
        else:
            u = lambda s: self._slice_mat(self.scal_upts_inb, s)
            uout = lambda s: self._slice_mat(self.scal_upts_outb, s)
            d = lambda s: self._slice_mat(self._scal_fpts, s)
            f = lambda s: self._slice_mat(self._vect_upts, s)
            uu = lambda s: self._slice_mat(self.scal_upts_inb, s)
            pts, npts = 'upts', self.nupts

            inject.append('fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];')
            inject.append('''
                tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);
                ''')

        # Mesh regions
        regions = self._mesh_regions

        print('external args and vals', extrnl_args, extrnl_vals)
        if 'curved' in regions:
            self.kernels['tdisf_curved'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[npts, regions['curved']],
                cdefs=self.cdefs, inject=inject,
                extrns=extrnl_args, **extrnl_vals,
                u=u('curved'), f=f('curved'),
                smats=self.smat_at(pts, 'curved'),
                uout=uout('curved'), rcpdjac=self.rcpdjac_at('upts', 'curved'),
                d=d('curved'), uu=uu('curved')
            )

        if 'linear' in regions:
            upts = getattr(self, pts)
            self.kernels['tdisf_linear'] = lambda: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[npts, regions['linear']],
                cdefs=self.cdefs, inject=inject,
                extrns=extrnl_args, **extrnl_vals,
                u=u('linear'), f=f('linear'),
                verts=self.ploc_at('linspts', 'linear'), upts=upts,
                uout=uout('linear'), rcpdjac=self.rcpdjac_at('upts', 'linear'),
                d=d('linear'), uu=uu('linear')
            )
