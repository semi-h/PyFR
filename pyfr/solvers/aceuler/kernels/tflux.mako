# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              uout='out fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              d='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              exec1='in fptr libxsmm_xfsspmdm_execute'
              exec2='in fptr libxsmm_xfsspmdm_execute'
              blockk1='in fptr vptr' blockk2='in fptr vptr'
              cleank1='in fptr vptr' cleank2='in fptr vptr'>

//              func1='in fptr gimmik_ptr'
//              func2='in fptr gimmik_ptr'>

    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp')};

    //printf("%p, %p \n", exec1, blockk1);
    //printf("%p, %p \n", exec2, blockk2);
    //printf("buffarr %p, d_v %p\n",(void*)&buffarr, (void*)&d_v);
    //    {('buffarr[({0}*_ny+_y)*SZ*4 + X_IDX_AOSOA({1}, {nvars})%SZ]'
    //      .format(i,j,nvars=nvars))} = {' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
    //                             .format(i, k, j)
    //                             for k in range(ndims))};
    // Transform the fluxes
    //'f[{i}][{j}] = {' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]''
% for i, j in pyfr.ndrange(ndims, nvars):
    //f[{i}][{j}] = {' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
        ${('buffarr[({0}*_ny+_y)*SZ*{nvars} + X_IDX_AOSOA({1}, {nvars})%(SZ*{nvars})]'
          .format(i,j,nvars=nvars))} = ${' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
                                 .format(i, k, j)
                                 for k in range(ndims))};
% endfor

//splithere

//d
% for i in range(nvars):
    uout[${i}] = -rcpdjac*uout[${i}];
% endfor

</%pyfr:kernel>
