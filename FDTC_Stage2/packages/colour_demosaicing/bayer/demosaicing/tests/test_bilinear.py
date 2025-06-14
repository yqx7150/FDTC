# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour_demosaicing.bayer.demosaicing.bilinear`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest

from colour import read_image

from colour_demosaicing import TESTS_RESOURCES_DIRECTORY
from colour_demosaicing.bayer import demosaicing_CFA_Bayer_bilinear

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['BAYER_DIRECTORY', 'TestDemosaicing_CFA_Bayer_bilinear']

BAYER_DIRECTORY = os.path.join(TESTS_RESOURCES_DIRECTORY, 'colour_demosaicing',
                               'bayer')


class TestDemosaicing_CFA_Bayer_bilinear(unittest.TestCase):
    """
    Defines :func:`colour_demosaicing.bayer.demosaicing.bilinear.\
demosaicing_CFA_Bayer_bilinear` definition unit tests methods.
    """

    def test_demosaicing_CFA_Bayer_bilinear(self):
        """
        Tests :func:`colour_demosaicing.bayer.demosaicing.bilinear.\
demosaicing_CFA_Bayer_bilinear` definition.
        """

        for pattern in ('RGGB', 'BGGR', 'GRBG', 'GBRG'):
            CFA = os.path.join(BAYER_DIRECTORY, 'Lighthouse_CFA_{0}.exr')
            RGB = os.path.join(BAYER_DIRECTORY, 'Lighthouse_Bilinear_{0}.exr')

            np.testing.assert_almost_equal(
                demosaicing_CFA_Bayer_bilinear(
                    read_image(str(CFA.format(pattern)))[..., 0], pattern),
                read_image(str(RGB.format(pattern))),
                decimal=7)


if __name__ == '__main__':
    unittest.main()
