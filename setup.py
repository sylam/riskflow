from setuptools import setup, find_packages
import os

with open('README.md') as f:
    long_description = f.read()

here = os.path.dirname(os.path.abspath(__file__))

version_ns = {}
with open(os.path.join(here, 'riskflow', '_version.py')) as f:
    exec(f.read(), {}, version_ns)

setup(
    name='riskflow',
    version=version_ns['__version__'],
    packages=find_packages(),
    url='https://github.com/sylam/riskflow',
    license='GPLv3',
    author='shuaib.osman',
    author_email='shuaib.osman@investec.co.za',
    description='An XVA quantitative library with AAD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=['numpy>=1.16.1', 'scipy>=1.2.2', 'pandas>=1.0', 'matplotlib>=3.0', 'torch>=1.11.0',
                      'python-markdown-math>=0.6'],
    extras_require={
        'interactive': ['jupyter'],
    },
    entry_points={
        'console_scripts': [
            'RF_Bootstrap = bootstrap:main',
            'RF_Batch = riskflow_batch:main',
        ]},
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6'],
)
