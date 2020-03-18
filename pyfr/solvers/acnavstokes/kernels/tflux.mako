# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.aceuler.kernels.flux'/>
<%include file='pyfr.solvers.acnavstokes.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              uout='out fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              d='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              f='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              func1='in fptr gimmik_ptr'
              func2='in fptr gimmik_ptr'>
    // Compute the flux (F = Fi + Fv)
    fpdtype_t ftemp[${ndims}][${nvars}];
    ${pyfr.expand('inviscid_flux', 'u', 'ftemp')};
    ${pyfr.expand('viscous_flux_add', 'u', 'f', 'ftemp')};
//uout
    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
                                 .format(i, k, j)
                                 for k in range(ndims))};
% endfor

//splithere

//d
% for i in enumerate(range(4)):
    uout[${i}] = -rcpdjac*uout[${i}];
% endfor

</%pyfr:kernel>
