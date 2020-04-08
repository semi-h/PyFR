# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='pcond' ndim='2'
              dtmarch='scalar fpdtype_t'
              rhs='inout customrow fpdtype_t[1][${str(nvars)}]'
              pmat='in customrow fpdtype_t[${str(ncp*nvars)}][${str(nvars)}]'
              piv='in customrow fpdtype_t[${str(ncp*nlines)}][1]'>

// _y goes up to nlines, loop over ncp for each line

    fpdtype_t rhst[${ncp*nvars}], rhsx[${ncp*nvars}];
    fpdtype_t alpha, div;
    int pi, i, j;

// forward substitution
% for i in range(ncp*nvars):
    alpha = ${' + '.join(['0'] +
        [('pmat[({ncp}*_y+{m}+({i}*{nvars}+{j})*{ncp}*{nlines})][{n}]'
         +'*rhst[{ix}]')
        .format(m=ix//ncp, n=ix%ncp, ncp=ncp, nvars=nvars, nlines=nlines,
                i=i//ncp, j=i%ncp, ix=ix)
        for ix in range(i)]
    )};
    pi = piv[${i}+_y*${ncp*nvars}][0];
    i = (pi%${ncp*nvars})/${ncp};
    j = (pi%${ncp*nvars})%${ncp};
    rhst[${i}] = rhs[${ncp}*_y+i][j] - alpha;


% endfor

// backward substitution
% for i in range(ncp*nvars-1, -1, -1):
    alpha = ${' + '.join(['0'] +
        [('pmat[({ncp}*_y+{m}+({i}*{nvars}+{j})*{ncp}*{nlines})][{n}]'
         +'*rhsx[{ix}]')
        .format(m=ix//ncp, n=ix%ncp, ncp=ncp, nvars=nvars, nlines=nlines,
                i=i//ncp, j=i%ncp, ix=ix)
        for ix in range(ncp*nvars-1, i, -1)]
    )};
    div = pmat[(${ncp}*_y+${i//ncp}+(${i//ncp*nvars+i%ncp})*${ncp}*${nlines})][${i%ncp}];
    rhsx[${i}] = (rhst[${i}] - alpha)/div;


% endfor

% for i, j in pyfr.ndrange(ncp, nvars):
    rhs[${ncp}*_y+${i}][${j}] = rhsx[${i*nvars+j}]/dtmarch;
% endfor


</%pyfr:kernel>
