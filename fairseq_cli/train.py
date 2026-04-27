#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

# Add the parent directory to the path to import train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import cli_main

if __name__ == '__main__':
    cli_main()