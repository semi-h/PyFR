# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              uout='out fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              d='in fpdtype_t[${str(nvars)}]'
              exec1='in fptr libxsmm_xfsspmdm_execute'
              exec2='in fptr libxsmm_xfsspmdm_execute'
              blockk1='in fptr vptr' blockk2='in fptr vptr'>
//              func1='in fptr gimmik_ptr'
//              func2='in fptr gimmik_ptr'>
    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    ${(f'buffarr[({i}*_ny+_y)*BLK_SZ*{nvars} + X_IDX_AOSOA({j}, {nvars})]')} =
        ${' + '.join(f'smats[{i}][{k}]*ftemp[{k}][{j}]' for k in range(ndims))};
% endfor

//splithere

//d
% for i in range(nvars):
    uout[${i}] = -rcpdjac*uout[${i}];
% endfor
</%pyfr:kernel>
