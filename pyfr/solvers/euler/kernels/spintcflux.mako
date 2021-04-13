# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>

<%pyfr:kernel name='spintcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              u='in fpdtype_t[${str(nvars)}]'
              d='out fpdtype_t[${str(nvars)}]'
              nfpts='in fpdtype_t'
              exec='in fptr libxsmm_xfsspmdm_execute'
              blockk='in fptr vptr'>

// do nothing here nfpts u d

//splithere

    // Perform the Riemann solve
    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'fn')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] =  magnl*fn[${i}];
    ur[${i}] = -magnl*fn[${i}];
% endfor
</%pyfr:kernel>
