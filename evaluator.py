"""
Author: Viet Lai
"""

from benchmarks import *
from transformers.utils.logging import get_logger, ERROR

logger = get_logger()
logger.setLevel(ERROR)

if __name__ == '__main__':
    # t = testsuit_for_XNLI()

    t = TestSuit('NI')
    t.register(BenchmarkerForNaturalInstruction(task='task003'))

    model = ProxyClient()

    t.test(model)
    t.print_report()
