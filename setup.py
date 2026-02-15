#!/usr/bin/env python3
"""
Zexus Programming Language - Setup Configuration
"""
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.install import install
import sys
import os

class PostInstallCommand(install):
    """Post-installation message pointing to documentation"""
    def run(self):
        install.run(self)
        print("\n" + "="*70)
        print("🎉 Zexus Programming Language installed successfully!")
        print("="*70)
        print("\n📚 DOCUMENTATION:")
        print("   Complete feature guide: https://github.com/Zaidux/zexus-interpreter/blob/main/docs/features/ADVANCED_FEATURES_IMPLEMENTATION.md")
        print("   Developer docs: https://github.com/Zaidux/zexus-interpreter/blob/main/src/README.md")
        print("   README: https://github.com/Zaidux/zexus-interpreter/blob/main/README.md")
        print("\n🚀 QUICK START:")
        print("   zx run program.zx      # Run a Zexus program")
        print("   zx repl                # Start interactive REPL")
        print("   zx init my-app         # Create new project")
        print("   zx --help              # Show all commands")
        print("\n💡 EXAMPLE PROGRAMS:")
        print("   Find examples at: tests/examples/")
        print("\n📖 For AI assistants: Full language spec and all features are")
        print("   documented at the links above. Zexus supports:")
        print("   - Policy-as-code (PROTECT/VERIFY/RESTRICT)")
        print("   - Blockchain contracts and transactions")
        print("   - Persistent memory and leak detection")
        print("   - Dependency injection and mocking")
        print("   - Reactive state management (WATCH)")
        print("   - 50+ built-in functions")
        print("\n" + "="*70 + "\n")

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = []
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [
            Extension(
                "zexus.vm.fastops",
                ["src/zexus/vm/fastops.pyx"],
            )
        ],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except Exception:
    ext_modules = []

ext_modules.append(
    Extension(
        "zexus.vm.cabi",
        ["src/zexus/vm/cabi.c"],
    )
)

ext_modules.append(
    Extension(
        "zexus.vm.native_runtime",
        ["src/zexus/vm/native_runtime.cpp"],
        language="c++",
    )
)

setup(
    name='zexus',
    version='1.7.2',
    author='Zaidux',
    author_email='devnull@example.com',
    description='A modern, security-first programming language with blockchain support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Zaidux/zexus-interpreter',
    project_urls={
        'Documentation': 'https://github.com/Zaidux/zexus-interpreter/blob/main/docs/',
        'Source': 'https://github.com/Zaidux/zexus-interpreter',
        'Bug Reports': 'https://github.com/Zaidux/zexus-interpreter/issues',
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'click>=8.0',
        'rich>=10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
        ],
        'blockchain': [
            'web3>=5.0',
            'eth-account>=0.5',
        ],
        'lsp': [
            'pygls>=1.0.0',
        ],
        'jit': [
            'llvmlite>=0.41.0',
            'Cython>=0.29',
        ],
        'network': [
            'websockets>=12.0',
            'aiohttp>=3.9',
            'httpx>=0.27',
        ],
    },
    entry_points={
        'console_scripts': [
            'zx=zexus.cli.main:cli',
            'zexus=zexus.cli.main:cli',
            'zpm=zexus.cli.zpm:cli',
            'zx-pypy=zexus.cli.main:cli',
        ]
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Interpreters',
        'Topic :: Software Development :: Compilers',
        'Topic :: Security',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='programming-language interpreter compiler blockchain smart-contracts security policy-as-code',
    zip_safe=False,
    ext_modules=ext_modules,
)
