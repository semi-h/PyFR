# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>
#include <stdio.h>

#define PYFR_ALIGN_BYTES ${alignb}
#define SOA_SZ ${soasz}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef void (*gimmik_ptr) (int, fpdtype_t *, int, fpdtype_t *, int, double, double);

// OpenMP static loop scheduling functions
<%include file='loop-sched'/>

${next.body()}
