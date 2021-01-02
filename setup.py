import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get('encoding', 'utf8')) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


setup(
    # Metadata
    name='shapecheck',
    version=find_version('shapecheck', '__init__.py'),
    author='Nicholas Vadivelu',
    author_email='nicholas.vadivelu@gmail.com',
    url='https://github.com/n2cholas/shapecheck',
    description='Framework-agnostic library for checking array shapes at runtime.',
    long_description_content_type='text/markdown',
    long_description=read('README.md'),
    license='MIT',
    # Package info
    packages=find_packages(exclude=(
        'tests',
        'tests.*',
    )),
    package_data={'shapecheck': ['py.typed']},
    zip_safe=True,
    install_requires=[],
)
