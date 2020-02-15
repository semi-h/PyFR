# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def rhs(self, t, uinbank, foutbank, xi=False):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        q1 << kernels['eles', 'disu_ext']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        q1 << kernels['eles', 'disu_int']()
        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        if ('iint', 'copy_fpts') in kernels:
            q1 << kernels['iint', 'copy_fpts']()
        q1 << kernels['iint', 'con_u']()
        q1 << kernels['bcint', 'con_u'](t=t)
        if ('eles', 'shocksensor') in kernels:
            q1 << kernels['eles', 'shocksensor']()
            q1 << kernels['mpiint', 'artvisc_fpts_pack']()
        if not xi:
            q1 << kernels['eles', 'tgradpcoru_upts']()
        else:
            q1 << kernels['eles', 'tgradpcoru_upts_xi']()
        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'con_u']()
        if not xi:
            q1 << kernels['eles', 'tgradcoru_upts_ext']()
        else:
            q1 << kernels['eles', 'tgradcoru_upts_xi_ext']()
        q1 << kernels['eles', 'gradcoru_upts_ext']()
        q1 << kernels['eles', 'gradcoru_fpts_ext']()
        q1 << kernels['mpiint', 'vect_fpts_pack']()
        if ('eles', 'shockvar') in kernels:
            q2 << kernels['mpiint', 'artvisc_fpts_send']()
            q2 << kernels['mpiint', 'artvisc_fpts_recv']()
            q2 << kernels['mpiint', 'artvisc_fpts_unpack']()

        runall([q1, q2])

        if not xi:
            q1 << kernels['eles', 'tgradcoru_upts_int']()
        else:
            q1 << kernels['eles', 'tgradcoru_upts_xi_int']()
        q1 << kernels['eles', 'gradcoru_upts_int']()
        q1 << kernels['eles', 'gradcoru_fpts_int']()
        if ('eles', 'gradcoru_qpts') in kernels:
            q1 << kernels['eles', 'gradcoru_qpts']()
        q1 << kernels['eles', 'tdisf']()
        if not xi:
            q1 << kernels['eles', 'tdivtpcorf']()
        else:
            q1 << kernels['eles', 'tdivtpcorf_xi']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux'](t=t)

        q2 << kernels['mpiint', 'vect_fpts_send']()
        q2 << kernels['mpiint', 'vect_fpts_recv']()
        q2 << kernels['mpiint', 'vect_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        if not xi:
            q1 << kernels['eles', 'tdivtconf']()
        else:
            q1 << kernels['eles', 'tdivtconf_xi']()
        if ('eles', 'tdivf_qpts') in kernels:
            q1 << kernels['eles', 'tdivf_qpts']()
            q1 << kernels['eles', 'negdivconf'](t=t)
            q1 << kernels['eles', 'divf_upts']()
        else:
            q1 << kernels['eles', 'negdivconf'](t=t)
        runall([q1])
