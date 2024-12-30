from setuptools import setup, find_packages

setup(
    name='riskmatrix',
    version='0.1.0',
    packages=find_packages(
        include=['riskmatrix']
    ),
    install_requires=[
        'pandas',
        'polars',
        'numpy',
    ],
)
