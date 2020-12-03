# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
    _nqueues = 2

    def rhs(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        q1 << kernels['eles', 'disu_ext']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        runall([q1])

        #q1 << kernels['eles', 'disu_int']()
        q1 << kernels['spiint', 'commpair_flux'](
            exec=self.disu_int_e_ptr, blockk=self.disu_int_b_ptr,
            cleank=self.disu_int_c_ptr
        )
        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        #q1 << kernels['eles', 'tdisf']()
        #q1 << kernels['eles', 'tdivtpcorf']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux'](t=t)

        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        q1 << kernels['eles', 'tdisf'](
            exec1=self.tdivtpcorf_e_ptr, exec2=self.tdivtconf_e_ptr,
            blockk1=self.tdivtpcorf_b_ptr, blockk2=self.tdivtconf_b_ptr,
            cleank1=self.tdivtpcorf_c_ptr, cleank2=self.tdivtconf_c_ptr
        )
        #    func1=self.tdivtpcorf_func_ptr, func2=self.tdivtconf_func_ptr
        #)
        #q1 << kernels['eles', 'tdivtconf']()
        #if ('eles', 'tdivf_qpts') in kernels:
        #    q1 << kernels['eles', 'tdivf_qpts']()
        #    q1 << kernels['eles', 'negdivconf'](t=t)
        #    q1 << kernels['eles', 'divf_upts']()
        #else:
        #    q1 << kernels['eles', 'negdivconf'](t=t)
        runall([q1])
