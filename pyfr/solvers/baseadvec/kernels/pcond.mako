# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='pcond' ndim='2'
              dtmarch='scalar fpdtype_t'
              rhs='inout customrow fpdtype_t[1][${str(nvars)}]'
              pmat='in customrow fpdtype_t[${str(ncp*nvars)}][${str(nvars)}]'>

// _y goes up to nlines, loop over ncp for each line

    fpdtype_t rhstemp[${ncp}][${nvars}];

% for i, j in pyfr.ndrange(ncp, nvars):
    rhstemp[${i}][${j}] = ${' + '.join(
        ('pmat[({ncp}*_y+{m}+({i}*{nvars}+{j})*{ncp}*{nlines})][{n}]'
         +'*rhs[{ncp}*_y+{m}][{n}]')
        .format(m=m, n=n, ncp=ncp, i=i, j=j, nvars=nvars, nlines=nlines)
        for m, n in pyfr.ndrange(ncp, nvars)
    )};
% endfor

% for i, j in pyfr.ndrange(ncp, nvars):
    rhs[${ncp}*_y+${i}][${j}] = rhstemp[${i}][${j}]/dtmarch;
% endfor
</%pyfr:kernel>
