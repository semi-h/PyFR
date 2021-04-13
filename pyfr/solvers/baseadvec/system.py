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

        if self.krnlgrouping == 'pairA':
            q1.enqueue(
                kernels['spiint', 'commpair_flux'],
                exec=self.disu_exec, blockk=self.disu_blkk
            )

            q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
            runall([q1])

            if ('eles', 'copy_soln') in kernels:
                q1.enqueue(kernels['eles', 'copy_soln'])
            q1.enqueue(kernels['iint', 'comm_flux'])
            q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

            q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

            runall([q1, q2])

            q1.enqueue(kernels['mpiint', 'comm_flux'])
            q1.enqueue(
                kernels['eles', 'tdisf_curved'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
                #exec1=self.tdivtpcorf_exec, exec2=self.tdivtconf_exec,
                #blockk1=self.tdivtpcorf_blkk, blockk2=self.tdivtconf_blkk
            )
            q1.enqueue(
                kernels['eles', 'tdisf_linear'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
                #exec1=self.tdivtpcorf_exec, exec2=self.tdivtconf_exec,
                #blockk1=self.tdivtpcorf_blkk, blockk2=self.tdivtconf_blkk
            )
            #func1=self.tdivtpcorf_func_ptr, func2=self.tdivtconf_func_ptr
            runall([q1])
        elif self.krnlgrouping == 'pairB':
            q1.enqueue(kernels['eles', 'disu'])

            q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
            runall([q1])

            if ('eles', 'copy_soln') in kernels:
                q1.enqueue(kernels['eles', 'copy_soln'])
            q1.enqueue(kernels['iint', 'comm_flux'])
            q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

            q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

            runall([q1, q2])

            q1.enqueue(kernels['mpiint', 'comm_flux'])
            q1.enqueue(
                kernels['eles', 'tdisf_curved'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
                #exec1=self.tdivtpcorf_exec, exec2=self.tdivtconf_exec,
                #blockk1=self.tdivtpcorf_blkk, blockk2=self.tdivtconf_blkk
            )
            q1.enqueue(
                kernels['eles', 'tdisf_linear'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
                #exec1=self.tdivtpcorf_exec, exec2=self.tdivtconf_exec,
                #blockk1=self.tdivtpcorf_blkk, blockk2=self.tdivtconf_blkk
            )
            #func1=self.tdivtpcorf_func_ptr, func2=self.tdivtconf_func_ptr
            runall([q1])
        elif self.krnlgrouping == 'pairC':
            q1.enqueue(
                kernels['eles', 'tdisf_curved'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
            )
            q1.enqueue(
                kernels['eles', 'tdisf_linear'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
            )

            if ('eles', 'copy_soln') in kernels:
                q1.enqueue(kernels['eles', 'copy_soln'])
            q1.enqueue(kernels['iint', 'comm_flux'])
            q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

            q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

            runall([q1, q2])

            q1.enqueue(kernels['mpiint', 'comm_flux'])
            #q1.enqueue(kernels['eles', 'tdivtconf'])
            q1.enqueue(
                kernels['eles', 'negdivconf'], t=t,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk
            )
            runall([q1])

        #else:
        elif self.krnlgrouping == 'pairD':
            q1.enqueue(kernels['eles', 'disu'])
            q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
            runall([q1])

            if ('eles', 'copy_soln') in kernels:
                q1.enqueue(kernels['eles', 'copy_soln'])
            q1.enqueue(
                kernels['eles', 'tdisf_curved'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
            )
            q1.enqueue(
                kernels['eles', 'tdisf_linear'],
                disu_exec=self.disu_exec,
                disu_blkk=self.disu_blkk,
                tdivtpcorf_exec=self.tdivtpcorf_exec,
                tdivtpcorf_blkk=self.tdivtpcorf_blkk,
                tdivtconf_exec=self.tdivtconf_exec,
                tdivtconf_blkk=self.tdivtconf_blkk,
                #qptsu_exec=self.qptsu_exec,
                #qptsu_blkk=self.qptsu_blkk,
            )
            #q1.enqueue(kernels['eles', 'tdisf_curved'])
            #q1.enqueue(kernels['eles', 'tdisf_linear'])
            q1.enqueue(kernels['eles', 'tdivtpcorf'])
            q1.enqueue(kernels['iint', 'comm_flux'])
            q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

            q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

            runall([q1, q2])

            q1.enqueue(kernels['mpiint', 'comm_flux'])
            q1.enqueue(kernels['eles', 'tdivtconf'])
            q1.enqueue(kernels['eles', 'negdivconf'], t=t)
            runall([q1])
