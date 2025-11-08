from setuptools import setup

setup(
    name='zexus-interpreter',
    version='0.1.0',
    description='Zexus language interpreter and hybrid compiler',
    author='Ziver-opensource',
    package_dir={'zexus': 'src/zexus'},
    packages=['zexus'],
    include_package_data=True,
    install_requires=[
        'click>=7.0',
        'rich>=9.0'
    ],
    entry_points={
        'console_scripts': [
            'zx = zexus.cli.main:cli'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
#!/usr/bin/env python3
from setuptools import setup

# This project uses setup.cfg for most metadata. Keep a minimal setup.py so
# users can install with `pip install -e .` in development mode.
if __name__ == '__main__':
    setup()
