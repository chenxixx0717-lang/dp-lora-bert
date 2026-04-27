#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages, Extension
# 注释掉这行：from torch.utils import cpp_extension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')


with open('README.md') as f:
    readme = f.read()


if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs

def config_fastbpe():
    try:
        from Cython.Build import cythonize
        # 返回正确的扩展，使用正确的文件路径
        return Extension(
            'fastBPE',
            sources=['fastbpe/fastBPE/fastBPE.pyx'],
            language='c++',
            extra_compile_args=extra_compile_args,
        )
    except ImportError:
        # 当Cython不可用时返回None
        return None

# 创建一个函数来延迟导入torch
def get_cpp_extension():
    try:
        import torch
        from torch.utils import cpp_extension
        return cpp_extension.CppExtension(
            'fairseq.libnat',
            sources=[
                'fairseq/clib/libnat/edit_dist.cpp',
            ],
            include_dirs=[
                torch.utils.cpp_extension.include_paths()[0],  # torch include path
            ],
            library_dirs=[
                torch.utils.cpp_extension.library_paths()[0],  # torch library path
            ],
            libraries=['torch', 'torch_cpu'],
            extra_compile_args=extra_compile_args + ['-DWITH_CUDA'] if torch.cuda.is_available() else extra_compile_args,
        )
    except ImportError:
        # 如果torch不可用，跳过这个扩展
        print("Warning: torch not available, skipping libnat extension")
        return None

# 创建一个自定义的BuildExtension类
class CustomBuildExtension:
    def __new__(cls):
        try:
            from torch.utils import cpp_extension
            return cpp_extension.BuildExtension
        except ImportError:
            from setuptools.command.build_ext import build_ext
            return build_ext

extensions = [
    Extension(
        'fairseq.libbleu',
        sources=[
            'fairseq/clib/libbleu/libbleu.cpp',
            'fairseq/clib/libbleu/module.cpp',
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'fairseq.data.data_utils_fast',
        sources=['fairseq/data/data_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'fairseq.data.token_block_utils_fast',
        sources=['fairseq/data/token_block_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

# 有条件地添加可选扩展
libnat_ext = get_cpp_extension()
if libnat_ext is not None:
    extensions.append(libnat_ext)

fastbpe_ext = config_fastbpe()
if fastbpe_ext is not None:
    extensions.append(fastbpe_ext)


# 移除test_suite选项，因为它已被弃用
setup(
    name='fairseq',
    version='0.8.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    url='https://github.com/pytorch/fairseq',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        'cffi',
        'cython',
        'numpy',
        'regex',
        'sacrebleu',
        'torch',
        'tqdm',
        'scipy',
    ],
    packages=find_packages(exclude=['scripts', 'tests']),
    ext_modules=extensions,
    # test_suite='tests',  # 移除这行，因为它已被弃用
    entry_points={
        'console_scripts': [
            'fairseq-eval-lm = fairseq_cli.eval_lm:cli_main',
            'fairseq-generate = fairseq_cli.generate:cli_main',
            'fairseq-interactive = fairseq_cli.interactive:cli_main',
            'fairseq-preprocess = fairseq_cli.preprocess:cli_main',
            'fairseq-score = fairseq_cli.score:main',
            'fairseq-train = fairseq_cli.train:cli_main',
            'fairseq-validate = fairseq_cli.validate:cli_main',
        ],
    },
    cmdclass={'build_ext': CustomBuildExtension()},
    zip_safe=False,
)
