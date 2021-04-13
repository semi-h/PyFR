# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:kernel name='tfluxlin' ndim='2'
              u='inout fpdtype_t[${str(nvars)}]'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'
              uout='out fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              d='inout fpdtype_t[${str(nvars)}]'
              disu_exec='in fptr libxsmm_xfsspmdm_execute'
              tdivtpcorf_exec='in fptr libxsmm_xfsspmdm_execute'
              tdivtconf_exec='in fptr libxsmm_xfsspmdm_execute'
              disu_blkk='in fptr vptr'
              tdivtconf_blkk='in fptr vptr'
              tdivtpcorf_blkk='in fptr vptr'>
              //qptsu_exec='in fptr libxsmm_xfsspmdm_execute'
              //qptsu_blkk='in fptr vptr'
              //uu='in fpdtype_t[${str(nvars)}]'
//              func1='in fptr gimmik_ptr'
//              func2='in fptr gimmik_ptr'>
//uu
    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp', 'p', 'v')};

    // Compute the S matrices
    fpdtype_t smats[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', 'smats', 'djac')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    //f[{i}][{j}] =
    ${(f'buffarr[({i}*_ny+_y)*BLK_SZ*{nvars} + X_IDX_AOSOA({j}, {nvars})]')} =
        ${' + '.join(f'smats[{i}][{k}]*ftemp[{k}][{j}]' for k in range(ndims))};
% endfor

//splithere

//d
% for i in range(nvars):
    uout[${i}] = -rcpdjac*uout[${i}];
% endfor
</%pyfr:kernel>
