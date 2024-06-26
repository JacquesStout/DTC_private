from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.9.0'
DESCRIPTION = 'Diffusion-images To Connectome processing toolbox'
LONG_DESCRIPTION = 'Set of tools for processing of diffusion images into connectome form using dipy methods and others'

# Setting up
setup(
    name="DTC",
    version=VERSION,
    url='https://github.com/JacquesStout/DTC_private',
    author="Jacques Andrew Stout",
    author_email="jacques.stout@duke.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(exclude=['tests']),
    zip_safe=False,
    install_requires=['antspyx==0.4.2','begins==0.9', 'Cython==3.0.6', 'dcor==0.6', 'dill==0.3.7', 'dipy==1.7.0',
                      'fury==0.9.0', 'matplotlib==3.8.1', 'nibabel==5.1.0', 'nipype==1.8.6', 'numpy==1.26.2',
                      'opencv_python==4.8.1.78', 'openpyxl==3.1.2', 'pandas==2.2.1', 'paramiko==3.3.1',
                      'scikit_image==0.22.0', 'scikit_learn==1.3.2', 'scipy==1.11.3',
                      'setuptools==58.1.0', 'vtk==9.3.0', 'xlrd==2.0.1', 'XlsxWriter==3.1.9'],
    keywords=['python', 'package', 'dipy', 'diffusion to connectomes'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers in Neurobiology",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'pyautodep=pyautodep.__main__:main'
        ]
    }
)
