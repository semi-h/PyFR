#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

import h5py

from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition.add_argument('np', help='number of partitions or a colon '
                              'delimited list of weights')
    ap_partition.add_argument('mesh', help='input mesh file')
    ap_partition.add_argument('solns', metavar='soln', nargs='*',
                              help='input solution files')
    ap_partition.add_argument('outd', help='output directory')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition.add_argument('-p', dest='partitioner', choices=partitioners,
                              help='partitioner to use')
    ap_partition.add_argument('-r', dest='rnumf', type=FileType('w'),
                              help='output renumbering file')
    ap_partition.add_argument('--popt', dest='popts', action='append',
                              default=[], metavar='key:value',
                              help='partitioner-specific option')
    ap_partition.add_argument('-t', dest='order', type=int, default=3,
                              help='target polynomial order; aids in '
                              'load-balancing mixed meshes')
    ap_partition.set_defaults(process=process_partition)

    # Color command
    ap_color = sp.add_parser('color', help='color --help')
    ap_color.add_argument('mesh', help='input mesh file')
    ap_color.add_argument('outd', help='output directory')
    ap_color.add_argument('-nc', dest='nminclr', type=int, default=4)
    ap_color.add_argument('-d', dest='dist', type=int, default=2)
    ap_color.add_argument('-t', dest='order', type=int, default=3,
                          help='target polynomial order; aids in '
                          'load-balancing mixed meshes')
    ap_color.add_argument('--align-xi', action='store_true')
    ap_color.set_defaults(process=process_color)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    ap_export.add_argument('outf', type=str, help='output file')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-d', '--divisor', type=int, default=0,
                           help='sets the level to which high order elements '
                           'are divided; output is linear between nodes, so '
                           'increased resolution may be required')
    ap_export.add_argument('-g', '--gradients', action='store_true',
                           help='compute gradients')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='output number precision; '
                           'defaults to single')
    ap_export.set_defaults(process=process_export)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('--backend', '-b', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('--progress', '-p', action='store_true',
                       help='show a progress bar')

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm()

    # Save to disk
    with h5py.File(args.outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Partitioner-specific options
    opts = dict(s.split(':', 1) for s in args.popts)

    # Create the partitioner
    if args.partitioner:
        part = get_partitioner(args.partitioner, pwts, order=args.order,
                               opts=opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner(name, pwts, order=args.order)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('No partitioners available')

    # Partition the mesh
    mesh, rnum, part_soln_fn = part.partition(NativeReader(args.mesh))

    # Prepare the solutions
    solnit = (part_soln_fn(NativeReader(s)) for s in args.solns)

    # Output paths/files
    paths = it.chain([args.mesh], args.solns)
    files = it.chain([mesh], solnit)

    # Iterate over the output mesh/solutions
    for path, data in zip(paths, files):
        # Compute the output path
        path = os.path.join(args.outd, os.path.basename(path.rstrip('/')))

        # Save to disk
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f[k] = v

    # Write out the renumbering table
    if args.rnumf:
        print('etype,pold,iold,pnew,inew', file=args.rnumf)

        for etype, emap in sorted(rnum.items()):
            for k, v in sorted(emap.items()):
                print(','.join(map(str, (etype, *k, *v))), file=args.rnumf)


def process_color(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    mesh = NativeReader(args.mesh)

    newm = dict()
    for dataset in mesh:
        newm[dataset] = mesh[dataset]

    import numpy as np
    # Preprocess mesh to have shorter distance aligned with ksi coordinate
    if args.align_xi:
        elemtype = ''
        for dataset in mesh:
            if dataset == 'spt_quad_p0' or dataset == 'spt_hex_p0':
                elemtype = dataset
                print('xi alignment will be applied to {0}'.format(elemtype))

        rlist = list()
        ccrlist = list()
        ndim = mesh[elemtype].shape[2]
        o = int(round(np.power(mesh[elemtype].shape[0], 1/ndim))) - 1
        refelem = np.zeros(((o+1)**ndim, ndim))

        def dist(n, m):
            return np.linalg.norm(n-m)

        for e in range(newm[elemtype].shape[1]):
            elem = newm[elemtype][:, e, :]
            dx = 0.5*(dist(elem[o], elem[0])
                      + dist(elem[o*(o+1)], elem[(o+1)*(o+1)-1]))
            dy = 0.5*(dist(elem[o], elem[(o+1)*(o+1)-1])
                      + dist(elem[0], elem[o*(o+1)]))

            if ndim == 2:
                if dx > dy:
                    # rotate element and add element to the list of rotated
                    rlist.append(e)

                    for j in range(o+1):
                        for k in range(o+1):
                            refelem[j*(o+1)+k] = elem[(k+1)*(o+1)-j-1]

                    newm['spt_quad_p0'][:, e, :] = refelem[:, :]
            elif ndim == 3:
                fn = (o+1)*(o+1)
                s = o*fn
                dz = 0.25*(dist(elem[0], elem[s])
                           + dist(elem[o], elem[o+s])
                           + dist(elem[o*(o+1)], elem[o*(o+1)+s])
                           + dist(elem[fn-1], elem[fn-1+s]))

                if dz > dy and dx > dy:
                    ccrlist.append(e)

                    for i in range(o+1):
                        for j in range(fn):
                            refelem[i*fn + j] = elem[i+j*(o+1)]

                    newm['spt_hex_p0'][:, e, :] = refelem[:, :]
                elif dy > dz and dx > dz:
                    rlist.append(e)

                    for i in range(fn):
                        for j in range(o+1):
                            refelem[i*(o+1)+j] = elem[i+j*fn]

                    newm['spt_hex_p0'][:, e, :] = refelem[:, :]

        print('Number of rotated elements', len(rlist), 'cc', len(ccrlist))

        rotface = [4, 0, 3, 5, 1, 2]
        rotccface = [1, 4, 5, 2, 0, 3]
        # Fix interface faces
        for dataset in newm:
            if 'con' in dataset:
                for e in newm[dataset][...].flat:
                    arr = e.astype('U4,i4,i1,i1')
                    if arr[0] == 'quad':
                        idx = np.searchsorted(rlist, arr[1])
                        if idx < len(rlist) and rlist[idx] == arr[1]:
                            e[2] = (e[2]-1) % 4
                            continue

                    if arr[0] == 'hex':
                        idx = np.searchsorted(rlist, arr[1])
                        if idx < len(rlist) and rlist[idx] == arr[1]:
                            e[2] = rotface[e[2]]
                            continue

                        idx = np.searchsorted(ccrlist, arr[1])
                        if idx < len(ccrlist) and ccrlist[idx] == arr[1]:
                            e[2] = rotccface[e[2]]
                            continue

    class GraphBuilder(BasePartitioner):
        def __init__(self, order):
            self.elewts = self.elewtsmap[min(order, max(self.elewtsmap))]

    graphbuilder = GraphBuilder(order=args.order)
    graph, vetimap = graphbuilder._construct_graph(newm)

    colors = set(range(args.nminclr))

    # vclr is the color array
    vclr = -np.ones(len(graph.vtab)-1, dtype=int)  # type: np.ndarray
    print('number of elements', len(vclr))
    print('number of colors', len(colors), colors)
    neighs = set()
    lclrs = set()
    clrwts = {c: 0 for c in colors}

    for i in range(len(graph.vtab[:-1])):
        neighs.update(set(graph.etab[graph.vtab[i]:graph.vtab[i+1]]))

        if args.dist == 2:
            for d2n in graph.etab[graph.vtab[i]:graph.vtab[i+1]]:
                neighs.update(set(graph.etab[graph.vtab[d2n]:graph.vtab[d2n+1]]))

        for neigh in neighs:
            lclrs.add(vclr[neigh]) if vclr[neigh] != -1 else None

        availclrs = colors.difference(lclrs)

        if len(availclrs) == 0:
            availclr = len(colors)
            colors.add(availclr)
            clrwts[availclr] = 0
        else:
            redclrwts = dict((k, clrwts[k]) for k in availclrs)
            availclr = min(redclrwts, key=redclrwts.get)

        vclr[i] = availclr
        clrwts[availclr] += 1 #graph.vwts[i]

        neighs.clear()
        lclrs.clear()

    # Check coloring
    for i in range(len(graph.vtab[:-1])):
        neighs.update(set(graph.etab[graph.vtab[i]:graph.vtab[i+1]]))

        if args.dist == 2:
            for d2n in graph.etab[graph.vtab[i]:graph.vtab[i+1]]:
                neighs.update(set(graph.etab[graph.vtab[d2n]:graph.vtab[d2n+1]]))

            neighs.remove(i)

        for neigh in neighs:
            lclrs.add(vclr[neigh])

        if vclr[i] in lclrs:
            print('Something wrong in the coloring :(')

        neighs.clear()
        lclrs.clear()

    print('number of colors:', len(colors), colors)
    print(clrwts)

    colorlist = [[] for i in range(len(colors))]

    for i, veti in enumerate(vetimap):
        colorlist[vclr[i]].append(veti)

    path = os.path.join(args.outd, os.path.basename(args.mesh.rstrip('/')))

    with h5py.File(path, 'w') as f:
        for k, v in newm.items():
            f[k] = v
        for c in colors:
            f['clr_{0}_p0'.format(c)] = np.array(colorlist[c], dtype='S4,i4')


def process_export(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    writer.write_out()


def _process_common(args, mesh, soln, cfg):
    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    # Ensure MPI is suitably cleaned up
    register_finalize_handler()

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and MPI.COMM_WORLD.rank == 0:
        pb = ProgressBar(solver.tstart, solver.tcurr, solver.tend)

        # Register a callback to update the bar after each step
        callb = lambda intg: pb.advance_to(intg.tcurr)
        solver.completed_step_handlers.append(callb)

    # Execute!
    solver.run()

    # Finalise MPI
    MPI.Finalize()


def process_run(args):
    _process_common(
        args, NativeReader(args.mesh), None, Inifile.load(args.cfg)
    )


def process_restart(args):
    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)

    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'] != mesh['mesh_uuid']:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'])

    _process_common(args, mesh, soln, cfg)


if __name__ == '__main__':
    main()
