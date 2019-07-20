# -*- coding: utf-8 -*-

from ast import parse
from distutils.sysconfig import get_python_lib
from functools import partial
from os import path, listdir
from platform import python_version_tuple

from setuptools import setup, find_packages

if python_version_tuple()[0] == '3':
    imap = map
    ifilter = filter
else:
    from itertools import imap, ifilter

if __name__ == '__main__':
    package_name = 'denet_glaucoma_screen'

    with open(path.join(package_name, '__init__.py')) as f:
        __author__, __version__ = imap(
            lambda buf: next(imap(lambda e: e.value.s, parse(buf).body)),
            ifilter(lambda line: line.startswith('__version__') or line.startswith('__author__'), f)
        )

    to_funcs = lambda *paths: (partial(path.join, path.dirname(__file__), package_name, *paths),
                               partial(path.join, get_python_lib(prefix=''), package_name, *paths))
    _data_join, _data_install_dir = to_funcs('_data')
    pre_model_join, pre_model_install_dir = to_funcs('pre_model')
    test_image_join, test_image_install_dir = to_funcs('test_image')

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        install_requires=['pyyaml'],
        test_suite=package_name + '.tests',
        packages=find_packages(),
        package_dir={package_name: package_name},
        data_files=[
            (_data_install_dir(), list(imap(_data_join, listdir(_data_join())))),
            (pre_model_install_dir(), list(imap(pre_model_join, listdir(pre_model_join())))),
            (test_image_install_dir(), list(imap(test_image_join, listdir(test_image_join()))))
        ]
    )
