# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import pytest


from innvestigate.utils.tests import dryrun

from innvestigate.analyzer import PatternNet
from innvestigate.analyzer import PatternAttribution


# todo: add again a traint/test case for mnist

###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternNet():

    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternNet():

    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__PatternAttribution():

    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "trivia.*:mnist.log_reg")


@pytest.mark.precommit
def test_precommit__PatternAttribution():

    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternAttribution(model, patterns=patterns)

    dryrun.test_analyzer(method, "mnist.*")


###############################################################################
###############################################################################
###############################################################################


@pytest.mark.fast
@pytest.mark.precommit
def test_fast__SerializePatternNet():

    def method(model):
        # enough for test purposes, only pattern application is tested here
        # pattern computation is tested separately.
        # assume that one dim weights are biases, drop them.
        patterns = [x for x in model.get_weights()
                    if len(x.shape) > 1]
        return PatternNet(model, patterns=patterns)

    dryrun.test_serialize_analyzer(method, "trivia.*:mnist.log_reg")