# -*- coding: utf-8 -*-

from itertools import chain, zip_longest

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        bodies = self.body.split('splithere')
        #print(bodies)
        if self.ndim == 1:
            core = [f'''
                   for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                   {{
                       #pragma omp simd
                       for (int _xj = 0; _xj < SOA_SZ; _xj++)
                       {{
                           {body}
                       }}
                   }}''' for body in bodies]
            clean = [f'''
                    for (int _xi = 0; _xi < _remi; _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {body}
                        }}
                    }}
                    for (int _xi = _remi, _xj = 0; _xj < _remj; _xj++)
                    {{
                        {body}
                    }}''' for body in bodies]
        else:
            core = [f'''
                   for (int _y = 0; _y < _ny; _y++)
                   {{
                       for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
                       {{
                           #pragma omp simd
                           for (int _xj = 0; _xj < SOA_SZ; _xj++)
                           {{
                               {body}
                           }}
                       }}
                   }}''' for body in bodies]
            clean = [f'''
                    for (int _y = 0; _y < _ny; _y++)
                    {{
                        for (int _xi = 0; _xi < _remi; _xi += SOA_SZ)
                        {{
                            #pragma omp simd
                            for (int _xj = 0; _xj < SOA_SZ; _xj++)
                            {{
                                {body}
                            }}
                        }}
                    }}
                    for (int _y = 0; _y < _ny; _y++)
                    {{
                        for (int _xi = _remi, _xj = 0; _xj < _remj; _xj++)
                        {{
                            {body}
                        }}
                    }}''' for body in bodies]

        porder = 4
        pn = porder+1
        nfpf = pn**2
        #tets
        #nupts, nfpts, nvar, nfpf = (pn*(pn+1)*(pn+2))//6, 4*pn*(pn+1)//2, 5, pn*(pn+1)//2
        print(nfpf)

        navstokes = True
        # pairA | pairB | pairC
        krnlgrouping = 'pairB'
        print('kernel grouping generator', krnlgrouping)

        if navstokes and (self.name == 'tflux' or self.name == 'tfluxlin'):
            # Navier-Stokes
            print('navstokes')
            corepack = f'''
                       fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                       fpdtype_t gbuff[nupts*ndims*BLK_SZ*nvars];
                       tgradpcoru_exec(tgradpcoru_blkk, u_v+ib*BLK_SZ*nvars*nupts, buffarr);
                       tgradcoru_exec(tgradcoru_blkk, d_v+ib*BLK_SZ*nvars*nfpts, buffarr);
                       {core[0]}
                       gradcoru_fpts0_exec(gradcoru_fpts0_blkk, gbuff, c_v+(ib*3)*BLK_SZ*nvars*nfpts);
                       gradcoru_fpts1_exec(gradcoru_fpts1_blkk, gbuff+BLK_SZ*nvars*nupts, c_v+(ib*3+1)*BLK_SZ*nvars*nfpts);
                       gradcoru_fpts2_exec(gradcoru_fpts2_blkk, gbuff+2*BLK_SZ*nvars*nupts, c_v+(ib*3+2)*BLK_SZ*nvars*nfpts);
                       //core[1]
                       tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                       '''
            cleanpack = f'''
                        fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                        fpdtype_t gbuff[nupts*ndims*BLK_SZ*nvars];
                        tgradpcoru_exec(tgradpcoru_blkk, u_v+ib*BLK_SZ*nvars*nupts, buffarr);
                        tgradcoru_exec(tgradcoru_blkk, d_v+ib*BLK_SZ*nvars*nfpts, buffarr);
                        {clean[0]}
                        gradcoru_fpts0_exec(gradcoru_fpts0_blkk, gbuff, c_v+(ib*3)*BLK_SZ*nvars*nfpts);
                        gradcoru_fpts1_exec(gradcoru_fpts1_blkk, gbuff+BLK_SZ*nvars*nupts, c_v+(ib*3+1)*BLK_SZ*nvars*nfpts);
                        gradcoru_fpts2_exec(gradcoru_fpts2_blkk, gbuff+2*BLK_SZ*nvars*nupts, c_v+(ib*3+2)*BLK_SZ*nvars*nfpts);
                        //clean[1]
                        tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                        '''
        elif (krnlgrouping == 'pairA' or krnlgrouping == 'pairB') and \
                (self.name == 'tflux' or self.name == 'tfluxlin'):
            print('tflux is here pairA or pairB')
            func1 = 'exec1(blockk1, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);'
            func2 = 'exec2(blockk2, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);'
            #buffarr = 'buffarr = (fpdtype_t*) calloc(64*3*SZ*4, sizeof(fpdtype_t)); //valid for p3 hex'
            #buffarr = '__attribute__((aligned(64))) fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            buffarr = 'fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            corepack = f'''
                       //qptsu_exec(qptsu_blkk, uu_v+ib*BLK_SZ*nvars*nupts, u_v+ib*BLK_SZ*nvars*nqpts);
                       //fpdtype_t buffarr[nqpts*ndims*BLK_SZ*nvars];
                       fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                       {core[0]}
                       tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                       tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);
                       //_nyy = nupts;
                       {core[1]}
                       '''
            cleanpack = f'''
                        //qptsu_exec(qptsu_blkk, uu_v+ib*BLK_SZ*nvars*nupts, u_v+ib*BLK_SZ*nvars*nqpts);
                        //fpdtype_t buffarr[nqpts*ndims*BLK_SZ*nvars];
                        fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                        {clean[0]}
                        tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                        tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);
                        //_nyy = nupts;
                        {clean[1]}
                        '''
        elif (krnlgrouping == 'pairC') and \
                (self.name == 'tflux' or self.name == 'tfluxlin'):
            print('tflux is here, pairC')
            func1 = 'exec1(blockk1, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);'
            func2 = 'exec2(blockk2, d_v+ib*BLK_SZ*nvars*nfpts, uout_v+ib*BLK_SZ*nvars*nupts);'
            #buffarr = 'buffarr = (fpdtype_t*) calloc(64*3*SZ*4, sizeof(fpdtype_t)); //valid for p3 hex'
            #buffarr = '__attribute__((aligned(64))) fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            buffarr = 'fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            corepack = f'''
                       {buffarr}
                       disu_exec(disu_blkk, u_v+ib*BLK_SZ*nvars*nupts, d_v+ib*BLK_SZ*nvars*nfpts);
                       {core[0]}
                       tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                       '''
            cleanpack = f'''
                        {buffarr}
                        disu_exec(disu_blkk, u_v+ib*BLK_SZ*nvars*nupts, d_v+ib*BLK_SZ*nvars*nfpts);
                        {clean[0]}
                        tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                        '''
        elif (navstokes or krnlgrouping == 'pairC') and self.name == 'negdivconf':
            corepack = f'''
                       tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                       {core[0]}
                       '''
            cleanpack = f'''
                        tdivtconf_exec(tdivtconf_blkk, d_v+ib*BLK_SZ*nvars*nfpts, tdivtconf_v+ib*BLK_SZ*nvars*nupts);
                        {clean[0]}
                        '''
        elif krnlgrouping == 'pairD' and \
            (self.name == 'tflux' or self.name == 'tfluxlin'):
            print('tflux is here pairD')
            #buffarr = 'buffarr = (fpdtype_t*) calloc(64*3*SZ*4, sizeof(fpdtype_t)); //valid for p3 hex'
            #buffarr = '__attribute__((aligned(64))) fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            buffarr = 'fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];'
            corepack = f'''
                       fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                       {core[0]}
                       tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                       //tdivtpcorf_exec(tdivtpcorf_blkk, f_v+ib*BLK_SZ*nvars*nupts*ndims, uout_v+ib*BLK_SZ*nvars*nupts);
                       '''
            cleanpack = f'''
                        //fpdtype_t buffarr[nupts*ndims*BLK_SZ*nvars];
                        {clean[0]}
                        //tdivtpcorf_exec(tdivtpcorf_blkk, buffarr, uout_v+ib*BLK_SZ*nvars*nupts);
                        //tdivtpcorf_exec(tdivtpcorf_blkk, f_v+ib*BLK_SZ*nvars*nupts*ndims, uout_v+ib*BLK_SZ*nvars*nupts);
                        '''
        elif self.name == 'spintcflux':
            print('spintcflux core')
            func2 = 'exec(blockk, u_v+ib*BLK_SZ*nvars*nupts, d_v+ib*BLK_SZ*nvars*nfpts);'
            corepack = f'''
                       {func2}
                       //begin
                       int begin, end;
                       if ( ib == 0 )
                           begin = 0;
                       else
                           begin = nfpts_v[ib-1]*Nfpf;
                       end = nfpts_v[ib]*Nfpf;
                       //printf("%d %d %d %f\\n", begin, end, ib, nfpts_v[ib]);
                       //printf("%d aoao\\n", ib);
                       //printf("%f \\n", nfpts_v[ib]);
                       //printf("%d \\n", nfpts_v[ib]);
                       //printf("%d %d \\n", begin, end);
                       if ( begin == end) printf("aaaa zeroo");
                       //for (int xi = begin; xi < end; xi++)
                       //{{
                       //    int _xi, _xj;
                       //    // this ib is different then the main loop ib
                       //    ib = xi/BLK_SZ;
                       //    //printf("%d ch\\n", ib);
                       //    _xi = xi%BLK_SZ;
                       //    //ib = 0;
                       //    //_xi = xi;
                       //    _xj = 0;
                       //    //printf("%d %d %d\\n", BLK_IDX, X_IDX, xi);
                       //    bodies[1]
                       //}}

                       //ib = i;
                       int nc;
                       nc = (end-begin)/SOA_SZ*SOA_SZ;
                       for (int xi = begin; xi < begin + nc; xi += SOA_SZ)
                       {{
                           #pragma omp simd
                           for (int xj = 0; xj < SOA_SZ; xj++)
                           {{
                               ib = (xi+xj)/BLK_SZ;
                               int _xi, _xj;
                               _xi = (xi+xj)%BLK_SZ;
                               _xj = 0;
                               {bodies[1]}
                           }}
                       }}
                       for (int xi = begin + nc, xj = 0; xj < end - xi; xj++)
                       {{
                           ib = (xi+xj)/BLK_SZ;
                           int _xi, _xj;
                           _xi = (xi+xj)%BLK_SZ;
                           _xj = 0;
                           {bodies[1]}
                       }}'''
            cleanpack = f'''
                        {func2}
                        int begin, end;
                        begin = nfpts_v[ib-1]*Nfpf;
                        end = nfpts_v[ib]*Nfpf;
                        if ( begin == end) printf("aaaa zeroo");
                        //for (int xi = begin; xi < end; xi++)
                        //{{
                            // this ib is different then the main loop ib
                        //    ib = xi/BLK_SZ;
                        //    int _xi, _xj;
                        //    _xi = xi%BLK_SZ;
                        //    _xj = 0;
                        //    bodies[1]
                        //}}
                        int nc;
                        nc = (end-begin)/SOA_SZ*SOA_SZ;
                        for (int xi = begin; xi < begin + nc; xi += SOA_SZ)
                        {{
                            #pragma omp simd
                            for (int xj = 0; xj < SOA_SZ; xj++)
                            {{
                                ib = (xi+xj)/BLK_SZ;
                                int _xi, _xj;
                                _xi = (xi+xj)%BLK_SZ;
                                _xj = 0;
                                {bodies[1]}
                            }}
                        }}
                        for (int xi = begin + nc, xj = 0; xj < end - xi; xj++)
                        {{
                            ib = (xi+xj)/BLK_SZ;
                            int _xi, _xj;
                            _xi = (xi+xj)%BLK_SZ;
                            _xj = 0;
                            {bodies[1]}
                        }}
                        '''
        else:
            corepack = f'{core[0]}'
            cleanpack = f'{clean[0]}'

        corepack = '\n'.join(
            [x for x in chain.from_iterable(zip_longest(self.inject, core))
             if x]
        )
        cleanpack = '\n'.join(
            [x for x in chain.from_iterable(zip_longest(self.inject, clean))
             if x]
        )

        #nyy = 'int _nyy = _ny;' if self.ndim == 2 else ''
        nyy = 'int _ny = _nyy;' if self.ndim == 2 else ''

        return f'''{self._render_spec()}
               {{
               int nci = _nx / BLK_SZ;
               int _remi = ((_nx % BLK_SZ) / SOA_SZ)*SOA_SZ;
               int _remj = (_nx % BLK_SZ) % SOA_SZ;
               #define X_IDX (_xi + _xj)
               #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
               {self.cdefs}
               #define Nfpf {nfpf}
               #define BLK_IDX ib*BLK_SZ
               #define BCAST_BLK(i, ld) ((i) % (ld) + ((i) / (ld))*(ld)*_ny)
               #pragma omp parallel for
               for (int ibo = 0; ibo < nci; ibo++)
               {{
                   int ib = ibo;
                   {nyy}
                   //_ny = _nyy;
                   {corepack}
               }}
               //if ( _remi != 0 && _remj != 0 )
               if ( !(_remi == 0 && _remj == 0) )
               {{
                   int ib = nci;
                   {nyy}
                   {cleanpack}
               }}
               #undef X_IDX
               #undef X_IDX_AOSOA
               #undef BLK_IDX
               #undef BCAST_BLK
               }}'''

    def ldim_size(self, name, *factor):
        return '*'.join(['BLK_SZ'] + [str(f) for f in factor])

    def needs_ldim(self, arg):
        return False

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.scalargs)

        # Function pointers
        kargs.extend(f'{sa.dtype} {sa.name}' for sa in self.fptrargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append(f'{va.dtype}* __restrict__ {va.name}_v')
                kargs.append(f'const int* __restrict__ {va.name}_vix')

                if va.ncdim == 2:
                    kargs.append(f'const int* __restrict__ {va.name}_vrstri')
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append(f'{const} {va.dtype}* __restrict__ {va.name}_v'
                             .strip())

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
