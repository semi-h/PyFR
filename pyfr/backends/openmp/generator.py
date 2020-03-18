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
                       func1(min(SZ*4,(tn-i*SZ)*4), f_v+i*SZ*4, ldf, uout_v+i*SZ*4, lduout, 1.0, 0.0);
                       //func(tn*4, f_v+i*SZ*4, ldf, uout_v+i*SZ*4, lduout);
                       //printf("%d %d %d\\n", _nx, ldf, tn);
                       //printf("%f \\n", u_v[0]);'''
            func2 = 'func2(min(SZ*4,(tn-i*SZ)*4), d_v+i*SZ*4, ldd, uout_v+i*SZ*4, lduout, 1.0, 1.0);'
        else:
            func1 = '//donothing func1${nvars}'
            func2 = '//donothing func2'

        if len(inner) == 2:
            inner2 = inner[1]
        else:
            inner2 = ''
        return '''{spec}
               {{
                   #define X_IDX (_xi + _xj)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
                   #define AOSOA_SZ 40
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
                   #pragma omp parallel for
                   for (int i = 0; i < lenAoAoSoA; i++)
                   {{
                       {inner}
                       {func1}
                       {func2}
                       {inner2}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(spec=self._render_spec(), inner=inner[0], inner2=inner2, func1=func1, func2=func2)

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
