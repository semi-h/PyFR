# -*- coding: utf-8 -*-
<%inherit file='base'/>

#include <stdalign.h>

// libxsmm prototype
typedef void (*libxsmm_xfsspmdm_execute)(void *, const fpdtype_t *,
                                         fpdtype_t *);

// gimmik prototype
typedef void (*gimmik_execute)(int, const fpdtype_t *, int, fpdtype_t *, int);

void
% if lib == 'xsmm':
% if krnl == 'disut':
batch_gemm(libxsmm_xfsspmdm_execute exec, void *blockk0, void *blockk1, void *blockk2, void *blockk3,
% else:
batch_gemm(libxsmm_xfsspmdm_execute exec, void *blockk,
% endif
% else:
batch_gemm(gimmik_execute exec, int bldim,
% endif
           int nblocks,
           const fpdtype_t *b, int bblocksz, fpdtype_t *c, int cblocksz)
{
    #pragma omp parallel for
    for (int ib = 0; ib < nblocks; ib++)
    % if lib == 'xsmm':
    % if krnl == 'disut':
    {
        fpdtype_t alignas(64) buff[cblocksz];
        fpdtype_t alignas(64) buff2[cblocksz];
        exec(blockk3, b + ib*bblocksz, buff);
        exec(blockk2, buff, buff2);
        exec(blockk1, buff2, buff);
        exec(blockk0, buff, c + ib*cblocksz);
        //exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
    }
    % elif krnl == 'disup':
    {
        fpdtype_t alignas(64) buff[cblocksz];
        fpdtype_t alignas(64) buff2[cblocksz];
        exec(blockk2, b + ib*bblocksz, buff);
        exec(blockk1, buff, buff2);
        exec(blockk0, buff2, c + ib*cblocksz);
        //exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
    }
    % else:
        exec(blockk, b + ib*bblocksz, c + ib*cblocksz);
    % endif
    % else:
        exec(bldim, b + ib*bblocksz, bldim, c + ib*cblocksz, bldim);
    % endif
}
