# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <omp.h>
#include <stdlib.h>
#include <tgmath.h>
#include <stdio.h>

#define PYFR_ALIGN_BYTES ${alignb}
#define SOA_SZ ${soasz}
#define AOSOA_SZ ${aosoasz}

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;
typedef void (*gimmik_ptr) (int, fpdtype_t *, int, fpdtype_t *, int, double, double);

typedef void (*libxsmm_xfsspmdm_execute)(void *, const fpdtype_t *, fpdtype_t *);

typedef void *vptr;

// OpenMP static loop scheduling functions
<%include file='loop-sched'/>

${next.body()}
