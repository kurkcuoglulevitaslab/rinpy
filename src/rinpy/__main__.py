# -*- coding: utf-8 -*-
"""
    __author__ = 'Zehra Sarica'
    __email__ = ['sarica16@itu.edu.tr','zehraacar559@gmail.com']
"""

import sys

from rinpy.network_comparator import main as compare_main
from rinpy.rin_process import main as process_main


def main():
    # `rinpy compare ...` compare CLI
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_main(sys.argv[2:])
        return
    # Default: RINProcess CLI
    process_main()


if __name__ == "__main__":
    main()
