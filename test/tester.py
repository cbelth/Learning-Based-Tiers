import unittest
from test_segment_inventory import TestSegmentInventory
from test_segment import TestSegment
from test_utils import TestUtils
from test_d2l import TestD2L

'''
A script to run all the test cases.
'''
# load test suites
test_seginv_suite = unittest.TestLoader().loadTestsFromTestCase(TestSegmentInventory)
test_segment_suite = unittest.TestLoader().loadTestsFromTestCase(TestSegment)
test_utils_suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
test_d2l_suite = unittest.TestLoader().loadTestsFromTestCase(TestD2L)
# combine the test suites
suites = unittest.TestSuite([
    test_seginv_suite,
    test_segment_suite,
    test_utils_suite,
    test_d2l_suite
])
# run the test suites
unittest.TextTestRunner(verbosity=2).run(suites)
