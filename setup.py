
from setuptools import setup

setup(
    name='smttask',
    version='0.2.0.dev0',
    description="Task wrapper for using Sumatra API",
    python_requires=">=3.7",

    author="Alexandre René",
    author_email="arene010@uottawa.ca",

    license='MIT',

    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3 :: Only',
      'Topic :: Scientific/Engineering'
    ],

    packages=['smttask'],

    install_requires=['pydantic',
                      'sumatra[git]>=0.8dev0',
                      #'psycopg2',
                      'mackelab-toolbox',
                      'parameters',
                      'click',
                      # Visualization dependencies
                      'networkx',
                      'holoviews',
                      'bokeh'
                      ],

    entry_points='''
        [console_scripts]
        smttask=smttask.ui:cli
    ''',
)
