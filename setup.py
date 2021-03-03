import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-cv-utils>=3.0.0',
    'wf-honeycomb-io>=0.3.0',
    'wf-video-io>=1.0.0',
    'pandas>=1.2.2',
    'numpy>=1.20.1',
    'opencv-python>=4.5.1'
]

# TEST_DEPENDENCIES = [
# ]

# DEVELOPMENT_DEPENDENCIES = [
# ]

# LOCAL_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-camera-calibration',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Support for calculating intrinsic and extrinsic camera calibration parameters using COLMAP and other tools',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-camera-calibration',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require = {
    #     'test': TEST_DEPENDENCIES,
    #     'development': DEVELOPMENT_DEPENDENCIES,
    #     'local': LOCAL_DEPENDENCIES
    # },
    # entry_points={
    #     "console_scripts": [
    #          "COMMAND_NAME = MODULE_PATH:METHOD_NAME"
    #     ]
    # },
    keywords=[],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
