# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict
import itertools as it
import re

from pyfr.inifile import Inifile
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclasses


class BaseSystem(object):
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Number of queues to allocate
    _nqueues = None

    # Nonce sequence
    _nonce_seq = it.count()

    def __init__(self, backend, rallocs, mesh, initsoln, nregs, cfg):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg

        # Obtain a nonce to uniquely identify this system
        nonce = str(next(self._nonce_seq))

        # Load the elements
        eles, elemap = self._load_eles(rallocs, mesh, initsoln, nregs, nonce)
        backend.commit()

        # Load colors
        self.clrmap = self._load_colors(rallocs, mesh, elemap)

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the banks, types, num DOFs and shapes of the elements
        self.ele_banks = list(eles.scal_upts_inb)
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = [(e.nupts, e.nvars, e.neles) for e in eles]
        self.ele_pmats = [e.pmat for e in eles]
        self.ele_ncp = [e.ncp for e in eles]

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        # I/O banks for the elements
        self.eles_scal_upts_inb = eles.scal_upts_inb
        self.eles_scal_upts_outb = eles.scal_upts_outb

        # Save the number of dimensions and field variables
        self.ndims = eles[0].ndims
        self.nvars = eles[0].nvars

        # Load the interfaces
        int_inters = self._load_int_inters(rallocs, mesh, elemap)
        mpi_inters = self._load_mpi_inters(rallocs, mesh, elemap)
        bc_inters = self._load_bc_inters(rallocs, mesh, elemap)
        backend.commit()

        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels(eles, int_inters, mpi_inters, bc_inters)
        backend.commit()

        # Save the BC interfaces, but delete the memory-intensive elemap
        self._bc_inters = bc_inters
        del bc_inters.elemap

    def _compute_int_offsets(self, rallocs, mesh):
        lhsprank = rallocs.prank
        intoffs = defaultdict(lambda: 0)

        for rhsprank in rallocs.prankconn[lhsprank]:
            interarr = mesh['con_p{0}p{1}'.format(lhsprank, rhsprank)]
            interarr = interarr[['f0', 'f1']].astype('U4,i4').tolist()

            for etype, eidx in interarr:
                intoffs[etype] = max(eidx + 1, intoffs[etype])

        return intoffs

    def _load_eles(self, rallocs, mesh, initsoln, nregs, nonce):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Look for and load each element type from the mesh
        elemap = OrderedDict()
        for f in mesh:
            m = re.match('spt_(.+?)_p{0}$'.format(rallocs.prank), f)
            if m:
                # Element type
                t = m.group(1)

                elemap[t] = self.elementscls(basismap[t], mesh[f], self.cfg)

        # Construct a proxylist to simplify collective operations
        eles = proxylist(elemap.values())

        # Set the initial conditions
        if initsoln:
            # Load the config and stats files from the solution
            solncfg = Inifile(initsoln['config'])
            solnsts = Inifile(initsoln['stats'])

            # Get the names of the conserved variables (fields)
            solnfields = solnsts.get('data', 'fields', '')
            currfields = ','.join(eles[0].convarmap[eles[0].ndims])

            # Ensure they match up
            if solnfields and solnfields != currfields:
                raise RuntimeError('Invalid solution for system')

            # Process the solution
            for etype, ele in elemap.items():
                soln = initsoln['soln_{0}_p{1}'.format(etype, rallocs.prank)]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            eles.set_ics_from_cfg()

        # Compute the index of first strictly interior element
        intoffs = self._compute_int_offsets(rallocs, mesh)

        # Allocate these elements on the backend
        for etype, ele in elemap.items():
            ele.set_backend(self.backend, nregs, nonce, intoffs[etype])

        return eles, elemap

    def _load_colors(self, rallocs, mesh, elemap):
        clrmap = OrderedDict()

        # Read colors
        for f in mesh:
            m = re.match(r'clr_(\d+?)_p{0}$'.format(rallocs.prank), f)
            if m:
                cn = m.group(1)

                clrmap[cn] = OrderedDict((k, []) for k in list(elemap))

                for ele in mesh[f]:
                    etype, elidx = ele.astype('U4,i4')
                    clrmap[cn][etype].append(elidx)

        return clrmap

    def _load_int_inters(self, rallocs, mesh, elemap):
        key = 'con_p{0}'.format(rallocs.prank)

        lhs, rhs = mesh[key].astype('U4,i4,i1,i1').tolist()
        int_inters = self.intinterscls(self.backend, lhs, rhs, elemap,
                                       self.cfg)

        # Although we only have a single internal interfaces instance
        # we wrap it in a proxylist for consistency
        return proxylist([int_inters])

    def _load_mpi_inters(self, rallocs, mesh, elemap):
        lhsprank = rallocs.prank

        mpi_inters = proxylist([])
        for rhsprank in rallocs.prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = mesh['con_p{0}p{1}'.format(lhsprank, rhsprank)]
            interarr = interarr.astype('U4,i4,i1,i1').tolist()

            mpiiface = self.mpiinterscls(self.backend, interarr, rhsmrank,
                                         rallocs, elemap, self.cfg)
            mpi_inters.append(mpiiface)

        return mpi_inters

    def _load_bc_inters(self, rallocs, mesh, elemap):
        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}

        bc_inters = proxylist([])
        for f in mesh:
            m = re.match('bcon_(.+?)_p{0}$'.format(rallocs.prank), f)
            if m:
                # Get the region name
                rgn = m.group(1)

                # Determine the config file section
                cfgsect = 'soln-bcs-%s' % rgn

                # Get the interface
                interarr = mesh[f].astype('U4,i4,i1,i1').tolist()

                # Instantiate
                bcclass = bcmap[self.cfg.get(cfgsect, 'type')]
                bciface = bcclass(self.backend, interarr, elemap, cfgsect,
                                  self.cfg)
                bc_inters.append(bciface)

        return bc_inters

    def _gen_queues(self):
        self._queues = [self.backend.queue() for i in range(self._nqueues)]

    def _gen_kernels(self, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(proxylist)

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provobjs = [eles, iint, mpiint, bcint]

        for pn, pobj in zip(provnames, provobjs):
            for kn, kgetter in it.chain(*pobj.kernels.items()):
                if not kn.startswith('_'):
                    kernels[pn, kn].append(kgetter())

    def rhs(self, t, uinbank, foutbank, xi=False):
        pass

    def get_preconditioner(self, u, dtmarch, adiag, lineimp):
        # construct the jacobian and invert before storing
        # get the solution from device to host
        # perturb and call rhs to construct to element jacobians
        # invert element jacobian in place
        import numpy as np

        eps = 1e-8

        base_soln = self.ele_scal_upts(u)

        self.rhs(0, u, u, xi=lineimp)
        base_derv = self.ele_scal_upts(u)

        self.restore_soln(u, base_soln)

        self.jacob = list()
        maxsize = 0

        for i, base_elemat in enumerate(base_soln):
            if not (self.ele_types[i] == 'quad' or
                    self.ele_types[i] == 'hex'):
                continue
            self.jacob.append(list())
            size = base_elemat.shape[0]*base_elemat.shape[1]

            for j in range(base_elemat.shape[2]):
                self.jacob[i].append(np.zeros((size, size)))

            if size > maxsize:
                maxsize = size

        from pyfr.mpiutil import get_comm_rank_root, get_mpi
        comm, rank, root = get_comm_rank_root()
        maxsize = comm.allreduce(maxsize, op=get_mpi('max'))

        for color in self.clrmap:
            for ncol in range(maxsize):
                for i, (base_elemat, eb) in enumerate(zip(base_soln,
                                                          self.ele_banks)):
                    if not (self.ele_types[i] == 'quad' or
                            self.ele_types[i] == 'hex'):
                        continue
                    size = base_elemat.shape[0]*base_elemat.shape[1]

                    if ncol >= size:
                        continue

                    elemat = base_elemat.copy()

                    i_nvar = ncol % base_elemat.shape[1]
                    i_u = ncol // base_elemat.shape[1]

                    for i_elem in self.clrmap[color][self.ele_types[i]]:
                        elemat[i_u, i_nvar, i_elem] += eps

                    eb[u].set(elemat)

                self.rhs(0, u, u, xi=lineimp)
                pert_derv = self.ele_scal_upts(u)
                self.restore_soln(u, base_soln)

                for i, base_elemat in enumerate(base_soln):
                    if not (self.ele_types[i] == 'quad' or
                            self.ele_types[i] == 'hex'):
                        continue
                    size = base_elemat.shape[0]*base_elemat.shape[1]

                    if ncol >= size:
                        continue

                    for i_elem in self.clrmap[color][self.ele_types[i]]:
                        diff = -(pert_derv[i][:, :, i_elem]
                                 - base_derv[i][:, :, i_elem])/eps
                        self.jacob[i][i_elem][:, ncol] = diff.reshape(-1)

        for i, base_elemat in enumerate(base_soln):
            if not (self.ele_types[i] == 'quad' or
                    self.ele_types[i] == 'hex'):
                continue
            size = base_elemat.shape[0]*base_elemat.shape[1]

            for i_elem in range(base_elemat.shape[2]):
                self.jacob[i][i_elem] *= adiag
                self.jacob[i][i_elem] += np.identity(size)/dtmarch
                self.jacob[i][i_elem] = np.linalg.inv(self.jacob[i][i_elem])

        for i, pmat in enumerate(self.ele_pmats):
            if pmat:
                ncp = self.ele_ncp[i]
                lsz = ncp*self.nvars

                # Allocate an empty array; ncp*nvars, nupts, nvars, neles
                nppmat = np.zeros((ncp*self.nvars, self.ele_shapes[i][0],
                                   self.nvars, self.ele_shapes[i][2]))

                nlines = int(self.ele_shapes[i][0]/ncp)

                for i_elem in range(self.ele_shapes[i][2]):
                    for line in range(nlines):
                        for i_ncp in range(ncp):
                            for i_nvar in range(self.nvars):
                                col = i_ncp*self.nvars+i_nvar
                                nppmat[
                                    col, line*ncp:(line+1)*ncp, :, i_elem
                                ] = self.jacob[i][i_elem][
                                        line*lsz+col, line*lsz:(line+1)*lsz
                                    ].reshape(-1, self.nvars)

                pmat.set(nppmat)

    def restore_soln(self, u, soln):
        for elemat, eb in zip(soln, self.ele_banks):
            eb[u].set(elemat)

    def precondition(self, r, dtmarch):
        derv = self.ele_scal_upts(r)

        for i, (elemat, eb) in enumerate(zip(derv, self.ele_banks)):
            if not (self.ele_types[i] == 'quad' or
                    self.ele_types[i] == 'hex'):
                continue
            s1, s2 = elemat.shape[0], elemat.shape[1]

            for i_elem in range(elemat.shape[2]):
                solnary = elemat[:, :, i_elem].reshape(-1)
                solnary = self.jacob[i][i_elem].dot(solnary)/dtmarch
                elemat[:, :, i_elem] = solnary.reshape((s1, s2))

            eb[r].set(elemat)

    def filt(self, uinoutbank):
        self.eles_scal_upts_inb.active = uinoutbank

        self._queues[0] % self._kernels['eles', 'filter_soln']()

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]
