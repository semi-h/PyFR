# -*- coding: utf-8 -*-

from pyfr.solvers.aceuler.elements import ACEulerElements
from pyfr.solvers.aceuler.inters import (ACEulerIntInters, ACEulerSpIntInters,
                                         ACEulerMPIInters, ACEulerBaseBCInters)
from pyfr.solvers.baseadvec import BaseAdvectionSystem


class ACEulerSystem(BaseAdvectionSystem):
    name = 'ac-euler'

    elementscls = ACEulerElements
    intinterscls = ACEulerIntInters
    spintinterscls = ACEulerSpIntInters
    mpiinterscls = ACEulerMPIInters
    bbcinterscls = ACEulerBaseBCInters
