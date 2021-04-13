# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              artvisc='in broadcast-col fpdtype_t'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              d='in fpdtype_t[${str(nvars)}]'
              c='out fpdtype_t[${str(nvars)}]'
              uout='out fpdtype_t[${str(nvars)}]'
              tgradcoru_exec='in fptr libxsmm_xfsspmdm_execute'
              tgradpcoru_exec='in fptr libxsmm_xfsspmdm_execute'
              gradcoru_fpts0_exec='in fptr libxsmm_xfsspmdm_execute'
              gradcoru_fpts1_exec='in fptr libxsmm_xfsspmdm_execute'
              gradcoru_fpts2_exec='in fptr libxsmm_xfsspmdm_execute'
              tdivtpcorf_exec='in fptr libxsmm_xfsspmdm_execute'
              tgradcoru_blkk='in fptr vptr'
              tgradpcoru_blkk='in fptr vptr'
              gradcoru_fpts0_blkk='in fptr vptr'
              gradcoru_fpts1_blkk='in fptr vptr'
              gradcoru_fpts2_blkk='in fptr vptr'
              tdivtpcorf_blkk='in fptr vptr'>
    fpdtype_t tmpgradu[${ndims}];

//d for scal_fpts, c for vec_ftps uout for scal_upts_out

% for j in range(nvars):
% for i in range(ndims):
    tmpgradu[${i}] = ${(f'buffarr[({i}*_ny+_y)*BLK_SZ*{nvars} + X_IDX_AOSOA({j}, {nvars})]')};
% endfor
% for i in range(ndims):
    ${(f'gbuff[({i}*_ny+_y)*BLK_SZ*{nvars} + X_IDX_AOSOA({j}, {nvars})]')} =
        rcpdjac*(${' + '.join(f'smats[{k}][{i}]*tmpgradu[{k}]'
                              for k in range(ndims))});
% endfor
% endfor

//splitnononohere

    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[${ndims}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp', 'p', 'v')};
    ${pyfr.expand('viscous_flux_buffadd', 'u', 'gbuff', 'ftemp')};
    ${pyfr.expand('artificial_viscosity_add', 'gbuff', 'ftemp', 'artvisc')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    ${(f'buffarr[({i}*_ny+_y)*BLK_SZ*{nvars} + X_IDX_AOSOA({j}, {nvars})]')} =
        ${' + '.join(f'smats[{i}][{k}]*ftemp[{k}][{j}]' for k in range(ndims))};
% endfor
</%pyfr:kernel>
