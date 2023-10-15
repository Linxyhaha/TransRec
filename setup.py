# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from distutils.core import setup, Extension
import os

extra_compile_args = ["-std=c++11", "-DNDEBUG", "-O3"]

extension = Extension(
    "generation_grounding.cpp_modules._fm_index",
    include_dirs=["generation_grounding/cpp_modules", os.path.expanduser("~/include")],
    libraries=["stdc++", "sdsl", "divsufsort", "divsufsort64"],
    library_dirs=[os.path.expanduser("~/lib")],
    sources=["generation_grounding/cpp_modules/fm_index.cpp", "generation_grounding/cpp_modules/fm_index.i"],
    swig_opts=["-I../include", "-c++"],
    language="c++11",
    extra_compile_args=extra_compile_args,
)

setup(
    name="TransRec",
    version="1.0",
    ext_modules=[extension],
)
