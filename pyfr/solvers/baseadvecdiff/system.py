# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def rhs(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        q1.enqueue(kernels['eles', 'disu'])
        q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
        runall([q1])

        if ('eles', 'copy_soln') in kernels:
            q1.enqueue(kernels['eles', 'copy_soln'])
        if ('iint', 'copy_fpts') in kernels:
            q1.enqueue(kernels['iint', 'copy_fpts'])
        q1.enqueue(kernels['iint', 'con_u'])
        q1.enqueue(kernels['bcint', 'con_u'], t=t)
        if ('eles', 'shocksensor') in kernels:
            q1.enqueue(kernels['eles', 'shocksensor'])
            q1.enqueue(kernels['mpiint', 'artvisc_fpts_pack'])
        #q1.enqueue(kernels['eles', 'tgradpcoru_upts'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

        runall([q1, q2])

        q1.enqueue(kernels['mpiint', 'con_u'])
        #q1.enqueue(kernels['eles', 'tgradcoru_upts'])
        #q1.enqueue(kernels['eles', 'gradcoru_upts_curved'])
        #q1.enqueue(kernels['eles', 'gradcoru_upts_linear'])
        #q1.enqueue(kernels['eles', 'gradcoru_fpts'])
        #q1.enqueue(kernels['mpiint', 'vect_fpts_pack'])  # take care of this for parallel runs
        if ('eles', 'shockvar') in kernels:
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_send'])
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_unpack'])

        runall([q1, q2])

        #if ('eles', 'gradcoru_qpts') in kernels:
        #    q1.enqueue(kernels['eles', 'gradcoru_qpts'])
        #if ('eles', 'qptsu') in kernels:
        #    q1.enqueue(kernels['eles', 'qptsu'])
        q1.enqueue(
            kernels['eles', 'tdisf_curved'], **self.krnlptrdict,
        )
        q1.enqueue( #kernels['eles', 'tdisf_linear'])
            kernels['eles', 'tdisf_linear'], **self.krnlptrdict,
        )

        # Try packing vec_fpts here
        q1.enqueue(kernels['mpiint', 'vect_fpts_pack'])  # parallel runs

        #q1.enqueue(kernels['eles', 'tdivtpcorf'])
        q1.enqueue(kernels['iint', 'comm_flux'])
        q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

        #q2.enqueue(kernels['mpiint', 'vect_fpts_send'])
        #q2.enqueue(kernels['mpiint', 'vect_fpts_recv'])
        #q2.enqueue(kernels['mpiint', 'vect_fpts_unpack'])

        runall([q1, q2])

        # Try send recv unpack here for vec_fpts
        q1.enqueue(kernels['mpiint', 'vect_fpts_send'])
        q1.enqueue(kernels['mpiint', 'vect_fpts_recv'])
        q1.enqueue(kernels['mpiint', 'vect_fpts_unpack'])

        q1.enqueue(kernels['mpiint', 'comm_flux'])
        #q1.enqueue(kernels['eles', 'tdivtconf'])
        q1.enqueue(
            kernels['eles', 'negdivconf'], t=t, **self.krnlptrdict,
            #tdivtconf_exec=self.tdivtconf_exec,
            #tdivtconf_blkk=self.tdivtconf_blkk
        )
        runall([q1])

    def rhsnotgood(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        q1.enqueue(kernels['eles', 'disu'])
        q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
        runall([q1])

        if ('eles', 'copy_soln') in kernels:
            q1.enqueue(kernels['eles', 'copy_soln'])
        if ('iint', 'copy_fpts') in kernels:
            q1.enqueue(kernels['iint', 'copy_fpts'])
        q1.enqueue(kernels['iint', 'con_u'])
        q1.enqueue(kernels['bcint', 'con_u'], t=t)
        if ('eles', 'shocksensor') in kernels:
            q1.enqueue(kernels['eles', 'shocksensor'])
            q1.enqueue(kernels['mpiint', 'artvisc_fpts_pack'])
        #q1.enqueue(kernels['eles', 'tgradpcoru_upts'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

        runall([q1, q2])

        q1.enqueue(kernels['mpiint', 'con_u'])
        #q1.enqueue(kernels['eles', 'tgradcoru_upts'])
        #q1.enqueue(kernels['eles', 'gradcoru_upts_curved'])
        #q1.enqueue(kernels['eles', 'gradcoru_upts_linear'])
        #q1.enqueue(kernels['eles', 'gradcoru_fpts'])
        #q1.enqueue(kernels['mpiint', 'vect_fpts_pack'])  # take care of this for parallel runs
        if ('eles', 'shockvar') in kernels:
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_send'])
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_recv'])
            q2.enqueue(kernels['mpiint', 'artvisc_fpts_unpack'])

        runall([q1, q2])

        #if ('eles', 'gradcoru_qpts') in kernels:
        #    q1.enqueue(kernels['eles', 'gradcoru_qpts'])
        #if ('eles', 'qptsu') in kernels:
        #    q1.enqueue(kernels['eles', 'qptsu'])
        q1.enqueue(
            kernels['eles', 'tdisf_curved'], **self.krnlptrdict,
        )
        q1.enqueue(kernels['mpiint', 'vect_fpts_pack'])  # parallel runs

        runall([q1])

        q1.enqueue( #kernels['eles', 'tdisf_linear'])
            kernels['eles', 'tdisf_linear'], **self.krnlptrdict,
        )

        q1.enqueue(kernels['iint', 'comm_flux'])
        q1.enqueue(kernels['bcint', 'comm_flux'], t=t)

        q2.enqueue(kernels['mpiint', 'vect_fpts_send'])
        print(q2.mpi_reqs, type(q2.mpi_reqs))
        q2.enqueue(kernels['mpiint', 'vect_fpts_recv'])
        print(q2.mpi_reqs, type(q2.mpi_reqs))
        q2.enqueue(kernels['mpiint', 'vect_fpts_unpack'])

        print(q2.mpi_reqs, type(q2.mpi_reqs))
        runall([q1, q2])
        ### Try packing vec_fpts here
        ##q1.enqueue(kernels['mpiint', 'vect_fpts_pack'])  # parallel runs

        #q1.enqueue(kernels['eles', 'tdivtpcorf'])

        #q2.enqueue(kernels['mpiint', 'vect_fpts_send'])
        #q2.enqueue(kernels['mpiint', 'vect_fpts_recv'])
        #q2.enqueue(kernels['mpiint', 'vect_fpts_unpack'])

        #runall([q1, q2])

        ### Try send recv unpack here for vec_fpts
        ##q1.enqueue(kernels['mpiint', 'vect_fpts_send'])
        ##q1.enqueue(kernels['mpiint', 'vect_fpts_recv'])
        ##q1.enqueue(kernels['mpiint', 'vect_fpts_unpack'])

        q1.enqueue(kernels['mpiint', 'comm_flux'])
        q1.enqueue(
            kernels['eles', 'negdivconf'], t=t, **self.krnlptrdict,
        )
        runall([q1])

    def compute_grads(self, t, uinbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank

        q1.enqueue(kernels['eles', 'disu'])
        q1.enqueue(kernels['mpiint', 'scal_fpts_pack'])
        runall([q1])

        if ('iint', 'copy_fpts') in kernels:
            q1.enqueue(kernels['iint', 'copy_fpts'])
        q1.enqueue(kernels['iint', 'con_u'])
        q1.enqueue(kernels['bcint', 'con_u'], t=t)
        q1.enqueue(kernels['eles', 'tgradpcoru_upts'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_send'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_recv'])
        q2.enqueue(kernels['mpiint', 'scal_fpts_unpack'])

        runall([q1, q2])

        q1.enqueue(kernels['mpiint', 'con_u'])
        q1.enqueue(kernels['eles', 'tgradcoru_upts'])
        q1.enqueue(kernels['eles', 'gradcoru_upts_curved'])
        q1.enqueue(kernels['eles', 'gradcoru_upts_linear'])

        runall([q1])
