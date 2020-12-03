# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class OpenMPKernelGenerator(BaseKernelGenerator):
    def render(self):
        bodies = self.body.split('splithere')
        #print(bodies)
        if self.ndim == 1:
            inner = ['''
                    //int cb, ce;
                    //loop_sched_1d(_nx, align, &cb, &ce);
                    //int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;
                    for (int _xi = i*SZ; _xi < min((i+1)*SZ,nci); _xi += SOA_SZ)
                    {{
                        #pragma omp simd
                        for (int _xj = 0; _xj < SOA_SZ; _xj++)
                        {{
                            {body}
                        }}
                    }}
                    if (nci < (i+1)*SZ)
                    {{
                        for (int _xi = min((i+1)*SZ,nci), _xj = 0; _xj < _nx - _xi;
                             _xj++)
                        {{
                            {body}
                        }}
                    }}'''.format(body=part) for part in bodies]
        else:
            inner = ['''
                    //int rb, re, cb, ce;
                    //loop_sched_2d(_ny, _nx, align, &rb, &re, &cb, &ce);
                    //int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;
                    for (int _y = 0; _y < _ny; _y++)
                    {{
                        for (int _xi = i*SZ; _xi < min((i+1)*SZ,nci);
                             _xi += SOA_SZ)
                        {{
                            #pragma omp simd
                            for (int _xj = 0; _xj < SOA_SZ; _xj++)
                            {{
                                {body}
                            }}
                        }}
                        if (nci < (i+1)*SZ)
                        {{
                            for (int _xi = min((i+1)*SZ,nci), _xj = 0;
                                 _xj < _nx - _xi; _xj++)
                            {{
                                {body}
                            }}
                        }}
                    }}'''.format(body=part) for part in bodies]

        if self.name == 'tflux':
            func1 = '''//printf("%f \\n", u_v[0]);
                       //func(1600, f_v, ldf, uout_v, lduout);
                       //int ncol = min(SZ*4, (i+1)*SZ-tn*SOA_SZ);
                       //func1(min(SZ*4,(tn-i*SZ)*4), f_v+i*SZ*4, ldf, uout_v+i*SZ*4, lduout, 1.0, 0.0);
                       //func1(min(SZ*4,(tn-i*SZ)*4), buffarr, SZ*4, uout_v+i*SZ*4, lduout, 1.0, 0.0);
                       if ( i != lenAoAoSoA-1 )
                       {
                           exec1(blockk1, buffarr, uout_v+i*SZ*NVAR);
                       }
                       else
                       {
                           exec1(cleank1, buffarr, uout_v+i*SZ*NVAR);
                       }
                       //exec1(blockk1, f_v+i*SZ*4, uout_v+i*SZ*4);
                       //func(tn*4, f_v+i*SZ*4, ldf, uout_v+i*SZ*4, lduout);
                       //printf("%d %d %d\\n", _nx, ldf, tn);
                       //printf("%f \\n", u_v[0]);'''
            func2 = '''//func2(min(SZ*4,(tn-i*SZ)*4), d_v+i*SZ*4, ldd, uout_v+i*SZ*4, lduout, 1.0, 1.0);
                       if ( i != lenAoAoSoA-1 )
                       {
                           exec2(blockk2, d_v+i*SZ*NVAR, uout_v+i*SZ*NVAR);
                       }
                       else
                       {
                           exec2(cleank2, d_v+i*SZ*NVAR, uout_v+i*SZ*NVAR);
                       }'''
            #buffarr = 'fpdtype_t fluxbuff[64*3][SZ*4]; //valid for p3 hex'
            #buffarr = 'buffarr = (fpdtype_t*) calloc(64*3*SZ*4, sizeof(fpdtype_t)); //valid for p3 hex'
            buffarr = '__attribute__((aligned(64))) fpdtype_t buffarr[Nu*3*SZ*NVAR];'
        elif self.name == 'spintcflux':
            inner[0] = ''
            func1 = '//donothing func1'
            func2 = '''
                       if ( i != lenAoAoSoA-1 )
                       {
                           exec(blockk, u_v+i*SZ*NVAR, d_v+i*SZ*NVAR);
                       }
                       else
                       {
                           exec(cleank, u_v+i*SZ*NVAR, d_v+i*SZ*NVAR);
                       }'''
            buffarr = '//donothing buffarr'
            inner[1] = '''
                       //begin
                       int begin, end;
                       if ( i == 0 )
                       {{
                           begin = 0;
                       }}
                       else
                       {{
                           begin = nfpts_v[i-1]*Nfpf;
                       }}
                       end = nfpts_v[i]*Nfpf;
                       //printf("%d aoao\\n", i);
                       //printf("%f \\n", nfpts_v[i]);
                       //printf("%d \\n", nfpts_v[i]);
                       //printf("%d %d \\n", begin, end);
                       int nc;
                       nc = (end-begin)/SOA_SZ*SOA_SZ;
                       for (int _xi = begin; _xi < begin + nc; _xi += SOA_SZ)
                       {{
                           #pragma omp simd
                           for (int _xj = 0; _xj < SOA_SZ; _xj++)
                           {{
                               {body}
                           }}
                       }}
                       for (int _xi = begin + nc, _xj = 0; _xj < end - _xi; _xj++)
                       {{
                           {body}
                       }}'''.format(body=bodies[1])
        else:
            func1 = '//donothing func1${nvars}'
            func2 = '//donothing func2'
            buffarr = '//donothing buffarr'

        if len(inner) == 2:
            inner2 = inner[1]
        else:
            inner2 = ''
        return '''{spec}
               {{
                   //printf(__func__);
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #define Nu {nupts}
                   #define Nfpf {nfpf}
                   #define NVAR {nvar}
                   #define SZ (AOSOA_SZ*SOA_SZ)
                   int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                   int nci = (_nx/SOA_SZ)*SOA_SZ;
                   int lenAoAoSoA = _nx/SOA_SZ;
                   if (_nx % SOA_SZ != 0)
                   {{
                       lenAoAoSoA += 1;
                   }}
                   int tn = lenAoAoSoA*SOA_SZ;
                   if (lenAoAoSoA % AOSOA_SZ != 0)
                   {{
                       lenAoAoSoA = lenAoAoSoA/AOSOA_SZ + 1;
                   }}
                   else
                   {{
                       lenAoAoSoA /= AOSOA_SZ;
                   }}
                   //fpdtype_t* buffarr;
                   //printf("lenaoaosoa %d \\n", lenAoAoSoA);
                   //printf("soasz %d %d %d\\n", SOA_SZ, AOSOA_SZ, _nx);
                   #pragma omp parallel for
                   for (int i = 0; i < lenAoAoSoA; i++)
                   {{
                       {buffarr}
                       {inner}
                       {func1}
                       {func2}
                       {inner2}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(
                    spec=self._render_spec(), inner=inner[0], inner2=inner2,
                    func1=func1, func2=func2, buffarr=buffarr, nvar=5,
                    nupts=64, nfpf=16
               )

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Function pointers
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.fptrargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append('{0.dtype}* __restrict__ {0.name}_v'.format(va))
                kargs.append('const int* __restrict__ {0.name}_vix'
                             .format(va))

                if va.ncdim == 2:
                    kargs.append('const int* __restrict__ {0.name}_vrstri'
                                 .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* __restrict__ {1.name}_v'
                             .format(const, va).strip())

                if self.needs_ldim(va):
                    kargs.append('int ld{0.name}'.format(va))

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
