
from setuptools import setup

setup(
    name='smttask',
    version='0.1.0dev',
    description="Task wrapper for using Sumatra API",

    author="Alexandre René",
    author_email="arene010@uottawa.ca",

    license='MIT',

    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3 :: Only',
      'Topic :: Scientific/Engineering'
    ],

    modules=['smttask.py'],

    install_requires=['attrdict',
                      'sumatra[git]',
                      'psycopg2',
                      'mackelab-toolbox'],
)
