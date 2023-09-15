# -*- coding: utf-8 -*-

import numpy as np

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'

    @staticmethod
    def grad_con_to_pri(cons, grad_cons, cfg):
        rho, *rhouvw = cons[:-1]
        grad_rho, *grad_rhouvw, grad_E = grad_cons
        
        # Divide momentum components by ρ
        uvw = [rhov/rho for rhov in rhouvw]

        # Velocity gradients
        # ∇(\vec{u}) = 1/ρ·[∇(ρ\vec{u}) - \vec{u} \otimes ∇ρ]
        grad_uvw = [(grad_rhov - v*grad_rho)/rho 
                    for grad_rhov, v in zip(grad_rhouvw, uvw)]

        # Pressure gradient
        # ∇p = (gamma - 1)·[∇E - 1/2*(\vec{u}·∇(ρ\vec{u}) - ρ\vec{u}·∇(\vec{u}))]
        gamma = cfg.getfloat('constants', 'gamma')
        grad_p = grad_E - 0.5*(np.einsum('ijk, iljk -> ljk', uvw, grad_rhouvw) +
                               np.einsum('ijk, iljk -> ljk', rhouvw, grad_uvw))
        grad_p *= (gamma - 1)

        return [grad_rho] + grad_uvw + [grad_p]

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.tfluxlin')

        # Handle shock capturing and Sutherland's law
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs,
            'shock_capturing': shock_capturing,
            'visc_corr': visc_corr
        }

        etype = self.basis.name
        # Set external arguments and values specific to the flux kernels
        extrnl_args = self._external_args.copy()
        extrnl_vals = self._external_vals.copy()
        extrnl_args['disu_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['tdivtpcorf_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        #extrnl_args['tdivtconf_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['disu_blkk'+'_'+etype] = 'in fptr vptr'
        #extrnl_args['tdivtconf_blkk'+'_'+etype] = 'in fptr vptr'
        extrnl_args['tdivtpcorf_blkk'+'_'+etype] = 'in fptr vptr'

        extrnl_args['tgradpcoru_upts_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['tgradcoru_upts_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_args['tgradpcoru_upts_blkk'+'_'+etype] = 'in fptr vptr'
        extrnl_args['tgradcoru_upts_blkk'+'_'+etype] = 'in fptr vptr'

        # Static code injection out of main loop
        inject = list()

        # Common arguments
        if 'flux' in self.antialias:
            u = lambda s: self._slice_mat(self._scal_qpts, s)
            f = lambda s: self._slice_mat(self._vect_qpts, s)
            uu = lambda s: self._slice_mat(self.scal_upts_inb, s)
            uout = lambda s: self._slice_mat(self.scal_upts_outb, s)
            d = lambda s: self._slice_mat(self._scal_fpts, s)
            c = lambda s: self._slice_mat(self._vect_fpts, s)
            pts, npts = 'qpts', self.nqpts

            extrnl_args['qptsu_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
            extrnl_args['qptsu_blkk'+'_'+etype] = 'in fptr vptr'
            extrnl_args['uu'] = f'in fpdtype_t[{self.nvars}]'

            tgradpcoru = f'tgradpcoru_upts_exec_{etype}(tgradpcoru_upts_blkk_{etype}, uu_v+ib*BLK_SZ*nvars*nupts, buffarr);'

            if self.basis.name == 'hex':
                for i in range(self.ndims):
                    extrnl_args['qptsu_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
                for i in range(self.ndims+1):
                    extrnl_args['tdivtpcorf_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
                qptsu = f'''
                    fpdtype_t alignas(64) buffqpts[nqpts*BLK_SZ*nvars];
                    fpdtype_t alignas(64) buffuq[nqpts*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk2_{etype}, uu_v+ib*BLK_SZ*nvars*nupts, buffuq);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, buffuq, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffuq);
                '''
                gradcoru_qpts = f'''
                    fpdtype_t alignas(64) buffqout[nqpts*ndims*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk2_{etype}, gbuff, buffqout);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, buffqout, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout);
                    qptsu_exec_{etype}(qptsu_blkk2_{etype}, gbuff+BLK_SZ*nvars*nupts, buffqout+BLK_SZ*nvars*nqpts);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, buffqout+BLK_SZ*nvars*nqpts, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout+BLK_SZ*nvars*nqpts);
                    qptsu_exec_{etype}(qptsu_blkk2_{etype}, gbuff+2*BLK_SZ*nvars*nupts, buffqout+2*BLK_SZ*nvars*nqpts);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, buffqout+2*BLK_SZ*nvars*nqpts, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout+2*BLK_SZ*nvars*nqpts);
                '''
                tdivtpcorf = f'''
                    //tdivtpcorf_exec_{etype}(tdivtpcorf_blkk_{etype}, buffqout, uout_v+ib*BLK_SZ*nvars*nupts);
                    fpdtype_t alignas(64) buffuout[nqpts*ndims*BLK_SZ*nvars];
                    fpdtype_t alignas(64) buffuout2[nqpts*ndims*BLK_SZ*nvars];
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk3_{etype}, buffqout, buffuout);
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk2_{etype}, buffuout, buffuout2);
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk1_{etype}, buffuout2, buffuout);
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk0_{etype}, buffuout, uout_v+ib*BLK_SZ*nvars*nupts);
                '''
            elif self.basis.name == 'pri':
                for i in range(2):
                    extrnl_args['qptsu_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
                for i in range(3):
                    extrnl_args['tdivtpcorf_blkk'+str(i)+'_'+etype] = 'in fptr vptr'

                qptsu = f'''
                    fpdtype_t alignas(64) buffqpts[nqpts*BLK_SZ*nvars];
                    fpdtype_t alignas(64) buffuq[nqpts*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, uu_v+ib*BLK_SZ*nvars*nupts, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffuq);
                '''
                gradcoru_qpts = f'''
                    fpdtype_t alignas(64) buffqout[nqpts*ndims*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, gbuff, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, gbuff+BLK_SZ*nvars*nupts, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout+BLK_SZ*nvars*nqpts);
                    qptsu_exec_{etype}(qptsu_blkk1_{etype}, gbuff+2*BLK_SZ*nvars*nupts, buffqpts);
                    qptsu_exec_{etype}(qptsu_blkk0_{etype}, buffqpts, buffqout+2*BLK_SZ*nvars*nqpts);
                '''
                tdivtpcorf = f'''
                    //tdivtpcorf_exec_{etype}(tdivtpcorf_blkk_{etype}, buffqout, uout_v+ib*BLK_SZ*nvars*nupts);
                    fpdtype_t alignas(64) buffuout[nqpts*ndims*BLK_SZ*nvars];
                    fpdtype_t alignas(64) buffuout2[nqpts*ndims*BLK_SZ*nvars];
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk2_{etype}, buffqout, buffuout);
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk1_{etype}, buffuout, buffuout2);
                    tdivtpcorf_exec_{etype}(tdivtpcorf_blkk0_{etype}, buffuout2, uout_v+ib*BLK_SZ*nvars*nupts);
                '''
            else:
                qptsu = f'''
                    fpdtype_t alignas(64) buffuq[nqpts*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk_{etype}, uu_v+ib*BLK_SZ*nvars*nupts, buffuq);
                '''
                gradcoru_qpts = f'''
                    fpdtype_t alignas(64) buffqout[nqpts*ndims*BLK_SZ*nvars];
                    qptsu_exec_{etype}(qptsu_blkk_{etype}, gbuff, buffqout);
                    qptsu_exec_{etype}(qptsu_blkk_{etype}, gbuff+BLK_SZ*nvars*nupts, buffqout+BLK_SZ*nvars*nqpts);
                    qptsu_exec_{etype}(qptsu_blkk_{etype}, gbuff+2*BLK_SZ*nvars*nupts, buffqout+2*BLK_SZ*nvars*nqpts);
                '''
                tdivtpcorf = f'tdivtpcorf_exec_{etype}(tdivtpcorf_blkk_{etype}, buffqout, uout_v+ib*BLK_SZ*nvars*nupts);'

        else:
            u = lambda s: self._slice_mat(self.scal_upts_inb, s)
            uu = lambda s: self._slice_mat(self.scal_upts_inb, s)
            uout = lambda s: self._slice_mat(self.scal_upts_outb, s)
            d = lambda s: self._slice_mat(self._scal_fpts, s)
            f = lambda s: self._slice_mat(self._vect_upts, s)
            c = lambda s: self._slice_mat(self._vect_fpts, s)
            pts, npts = 'upts', self.nupts

            tgradpcoru = f'tgradpcoru_upts_exec_{etype}(tgradpcoru_upts_blkk_{etype}, u_v+ib*BLK_SZ*nvars*nupts, buffarr);'
            gradcoru_qpts = 'fpdtype_t alignas(64) buffout[nupts*ndims*BLK_SZ*nvars];'
            qptsu = ''
            #tdivtpcorf = 'tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);'
            tdivtpcorf = f'tdivtpcorf_exec_{etype}(tdivtpcorf_blkk_{etype}, buffout, uout_v+ib*BLK_SZ*nvars*nupts);'

        if 'surf-flux' in self.antialias and self.basis.name == 'hex':
            for i in range(self.ndims+1):
                extrnl_args['tgradcoru_upts_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
                extrnl_args['disu_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
            tgradcoru = f'''
                fpdtype_t alignas(64) g2buff[nupts*ndims*BLK_SZ*nvars];
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk3_{etype}, c_v+ib*ndims*BLK_SZ*nvars*nfpts, gbuff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk2_{etype}, gbuff, g2buff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk1_{etype}, g2buff, gbuff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk0_{etype}, gbuff, buffarr);
            '''
            tgradcoru = f'''
                fpdtype_t alignas(64) gqbuff[nqpts*ndims*BLK_SZ*nvars];
                fpdtype_t alignas(64) gqbuff2[nqpts*ndims*BLK_SZ*nvars];
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk3_{etype}, c_v+ib*ndims*BLK_SZ*nvars*nfpts, gqbuff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk2_{etype}, gqbuff, gqbuff2);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk1_{etype}, gqbuff2, gqbuff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk0_{etype}, gqbuff, buffarr);
            '''
            #tgradcoru = 'tgradcoru_upts_exec(tgradcoru_upts_blkk, c_v+ib*ndims*BLK_SZ*nvars*nfpts, buffarr);'
            gradcoru_fpts = f'''
                fpdtype_t alignas(64) fptsbuff[nfpts*BLK_SZ*nvars];
                fpdtype_t alignas(64) fptsbuff2[nfpts*BLK_SZ*nvars];
                disu_exec_{etype}(disu_blkk3_{etype}, gbuff, fptsbuff);
                disu_exec_{etype}(disu_blkk2_{etype}, fptsbuff, fptsbuff2);
                disu_exec_{etype}(disu_blkk1_{etype}, fptsbuff2, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk3_{etype}, gbuff+BLK_SZ*nvars*nupts, fptsbuff);
                disu_exec_{etype}(disu_blkk2_{etype}, fptsbuff, fptsbuff2);
                disu_exec_{etype}(disu_blkk1_{etype}, fptsbuff2, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims+1)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk3_{etype}, gbuff+2*BLK_SZ*nvars*nupts, fptsbuff);
                disu_exec_{etype}(disu_blkk2_{etype}, fptsbuff, fptsbuff2);
                disu_exec_{etype}(disu_blkk1_{etype}, fptsbuff2, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims+2)*BLK_SZ*nvars*nfpts);
            '''
        elif 'surf-flux' in self.antialias and self.basis.name == 'pri':
            for i in range(3):
                extrnl_args['tgradcoru_upts_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
                extrnl_args['disu_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
            tgradcoru = f'''
                fpdtype_t alignas(64) gqbuff[nqpts*ndims*BLK_SZ*nvars];
                fpdtype_t alignas(64) gqbuff2[nqpts*ndims*BLK_SZ*nvars];
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk2_{etype}, c_v+ib*ndims*BLK_SZ*nvars*nfpts, gqbuff);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk1_{etype}, gqbuff, gqbuff2);
                tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk0_{etype}, gqbuff2, buffarr);
            '''
            #tgradcoru = 'tgradcoru_upts_exec(tgradcoru_upts_blkk, c_v+ib*ndims*BLK_SZ*nvars*nfpts, buffarr);'
            gradcoru_fpts = f'''
                fpdtype_t alignas(64) fptsbuff[nfpts*BLK_SZ*nvars];
                disu_exec_{etype}(disu_blkk2_{etype}, gbuff, c_v+(ib*ndims)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk1_{etype}, c_v+(ib*ndims)*BLK_SZ*nvars*nfpts, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk2_{etype}, gbuff, c_v+(ib*ndims+1)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk1_{etype}, c_v+(ib*ndims+1)*BLK_SZ*nvars*nfpts, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims+1)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk2_{etype}, gbuff, c_v+(ib*ndims+2)*BLK_SZ*nvars*nfpts);
                disu_exec_{etype}(disu_blkk1_{etype}, c_v+(ib*ndims+2)*BLK_SZ*nvars*nfpts, fptsbuff);
                disu_exec_{etype}(disu_blkk0_{etype}, fptsbuff, c_v+(ib*ndims+2)*BLK_SZ*nvars*nfpts);
            '''
        else:
            tgradcoru = f'tgradcoru_upts_exec_{etype}(tgradcoru_upts_blkk_{etype}, c_v+ib*ndims*BLK_SZ*nvars*nfpts, buffarr);'
            #gradcoru_fpts = '''
            #    disu_exec(disu_blkk, gbuff, c_v+(ib*ndims)*BLK_SZ*nvars*nfpts);
            #    disu_exec(disu_blkk, gbuff+BLK_SZ*nvars*nupts, c_v+(ib*ndims+1)*BLK_SZ*nvars*nfpts);
            #    disu_exec(disu_blkk, gbuff+2*BLK_SZ*nvars*nupts, c_v+(ib*ndims+2)*BLK_SZ*nvars*nfpts);
            #'''
            gradcoru_fpts = ''
            for i in range(self.ndims):
                gradcoru_fpts += f'''disu_exec_{etype}(disu_blkk_{etype}, gbuff+{i}*BLK_SZ*nvars*nupts, c_v+(ib*ndims+{i})*BLK_SZ*nvars*nfpts);\n'''

        inject.append(f'''
            fpdtype_t alignas(64) buffarr[nupts*ndims*BLK_SZ*nvars];
            fpdtype_t alignas(64) gbuff[nupts*ndims*BLK_SZ*nvars];
            //for (int ij = 0; ij < nupts*BLK_SZ*nvars; ij++) printf("%d u_v %d %d %f \\n", nupts, ib, ij, u_v[ij]);
            //for (int ij = 0; ij < nfpts*BLK_SZ*nvars; ij++) printf("%d c_v %d %d %f \\n", nfpts, ib, ij, c_v[ij]);
            {tgradpcoru}
            //for (int ij = 0; ij < nupts*ndims*BLK_SZ*nvars; ij++) printf("%d buffarr0 %d %d %f\\n", nupts, ib, ij, buffarr[ij]);
            {tgradcoru}
            //for (int ij = 0; ij < nupts*ndims*BLK_SZ*nvars; ij++) printf("%d buffarr %d %d %f\\n", nupts, ib, ij, buffarr[ij]);
            _ny = nupts;
        ''')
        inject.append(f'''
            //for (int ij = 0; ij < nupts*ndims*BLK_SZ*nvars; ij++) printf("%d gbuff buffarr %d %d %f %f\\n", nupts, ib, ij, gbuff[ij], buffarr[ij]);
            {gradcoru_fpts}
            {qptsu}
            {gradcoru_qpts}
            _ny = n{pts};
        ''')
        inject.append(f'''
            {tdivtpcorf}
        ''')

        av = self.artvisc

        # Mesh regions
        regions = self._mesh_regions
        print('regions', regions)

        tplargs['pts'] = pts

        if 'curved' in regions:
            self.kernels['tdisf_curved'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[npts, regions['curved']],
                cdefs=self.cdefs, inject=inject,
                extrns=extrnl_args, **extrnl_vals,
                u=u('curved'), f=f('curved'),
                artvisc=self._slice_mat(av, 'curved') if av else None,
                smats=self.smat_at(pts, 'curved'),
                uout=uout('curved'), rcpdjac=self.rcpdjac_at('upts', 'curved'),
                d=d('curved'), c=c('curved'), uu=uu('curved')
            )

        if 'linear' in regions:
            upts = getattr(self, pts)
            self.kernels['tdisf_linear'] = lambda: self._be.kernel(
                'tfluxlin', tplargs=tplargs, dims=[npts, regions['linear']],
                cdefs=self.cdefs, inject=inject,
                extrns=extrnl_args, **extrnl_vals,
                u=u('linear'), f=f('linear'),
                artvisc=self._slice_mat(av, 'linear') if av else None,
                verts=self.ploc_at('linspts', 'linear'), upts=upts,
                uout=uout('linear'), d=d('linear'), c=c('linear'),
                uu=uu('linear')
            )

        # Source term kernel arguments
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        extrnl_argsneg = self._external_args.copy()
        extrnl_argsneg['tdivtconf_exec'+'_'+etype] = 'in fptr libxsmm_xfsspmdm_execute'
        extrnl_argsneg['tdivtconf_blkk'+'_'+etype] = 'in fptr vptr'
        injectneg = list()

        if 'surf-flux' in self.antialias and self.basis.name == 'hex':
            for i in range(self.ndims+1):
                extrnl_argsneg['tdivtconf_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
            injectneg.append(f'''
                //tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                fpdtype_t alignas(64) buffuout[nqpts*ndims*BLK_SZ*nvars];
                fpdtype_t alignas(64) buffuout2[nqpts*ndims*BLK_SZ*nvars];
                tdivtconf_exec_{etype}(tdivtconf_blkk3_{etype}, d_v+ib*BLK_SZ*nvars*nfpts, buffuout);
                tdivtconf_exec_{etype}(tdivtconf_blkk2_{etype}, buffuout, buffuout2);
                tdivtconf_exec_{etype}(tdivtconf_blkk1_{etype}, buffuout2, buffuout);
                tdivtconf_exec_{etype}(tdivtconf_blkk0_{etype}, buffuout, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                ''')
        elif 'surf-flux' in self.antialias and self.basis.name == 'pri':
            for i in range(3):
                extrnl_argsneg['tdivtconf_blkk'+str(i)+'_'+etype] = 'in fptr vptr'
            injectneg.append(f'''
                //tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                fpdtype_t alignas(64) buffuout[nqpts*ndims*BLK_SZ*nvars];
                fpdtype_t alignas(64) buffuout2[nqpts*ndims*BLK_SZ*nvars];
                tdivtconf_exec_{etype}(tdivtconf_blkk2_{etype}, d_v+ib*BLK_SZ*nvars*nfpts, buffuout);
                tdivtconf_exec_{etype}(tdivtconf_blkk1_{etype}, buffuout, buffuout2);
                tdivtconf_exec_{etype}(tdivtconf_blkk0_{etype}, buffuout2, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                ''')
        else:
            injectneg.append(f'tdivtconf_exec_{etype}(tdivtconf_blkk_{etype}, d_v+ib*BLK_SZ*nvars*nfpts, tdivtconf_v+ib*BLK_SZ*nvars*nupts);')

        self.kernels['negdivconf'] = lambda: self._be.kernel(
            'negdivconf', tplargs=srctplargs,
            cdefs=self.cdefs, inject=injectneg,
            extrns=extrnl_argsneg,
            dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
            rcpdjac=self.rcpdjac_at('upts'),
            d=self._scal_fpts
        )
