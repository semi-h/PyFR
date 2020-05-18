# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='localalpha' ndim='2'
              au='scalar fpdtype_t' ao='scalar fpdtype_t'
              ru='in fpdtype_t[${str(nvars)}]'
              re='in fpdtype_t[${str(nvars)}]'
              ro='in fpdtype_t[${str(nvars)}]'
              negdivconf='inout fpdtype_t[${str(nvars)}]'
              sol='in fpdtype_t[${str(nvars)}]'
              dtau_upts ='inout fpdtype_t[${str(nvars)}]'>

// au, au overshoot and undershoot parameters

% for i in range(nvars):
    if ((ru[${i}] < re[${i}]) and (ru[${i}] < ro[${i}]))
    {
        negdivconf[${i}] = sol[${i}] + negdivconf[${i}]*au;
        dtau_upts[${i}] *= au;
    }
    else if ((ro[${i}] < re[${i}]) and (ro[${i}] < ru[${i}]))
    {
        negdivconf[${i}] = sol[${i}] + negdivconf[${i}]*ao;
        dtau_upts[${i}] *= ao;
    }
    else
    {
        negdivconf[${i}] = sol[${i}] + negdivconf[${i}];
    }
% endfor
</%pyfr:kernel>
