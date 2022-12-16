from setuptools import setup, find_packages

# https://stackoverflow.com/questions/45114076/python-setuptools-using-scripts-keyword-in-setup-py
setup(name='MetSeg',
      version='1.0',
      description='Metastases Segmentation',
      url='Insert Later',
      python_requires='>=3.8',
      author='Jon André Ottesen',
      author_email='jonakri@uio.no',
      license='Apache 2.0',
      install_requires=[
      ],
      scripts=['met-seg'],
      packages=find_packages(include=['m_seg']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )

