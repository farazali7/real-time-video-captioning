from setuptools import setup, find_packages

"""
The setup.py file is to install the packages in development mode.
This can be done by running the following command in the terminal:
>>> python setup.py build develop --user
"""

setup(
    name = 'real-time-video-captioning',
    version = '0.1',
    description = """
    Real-time video captioning. Course project for Visual and Mobile Computing Systems Class.
    """,
    install_requires = open("requirements.txt").read().splitlines(),
    packages = find_packages(),
 )