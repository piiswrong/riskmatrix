from setuptools import setup, find_packages

setup(
    name='quantlib',
    version='0.1.0',
    packages=find_packages(
        include=['quantlib']
    ),
    install_requires=[
        'pandas',
        'polars',
        'numpy',
    ],
)
