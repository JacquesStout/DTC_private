from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.6.0'
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
    install_requires=['antspyx==0.3.7','begins==0.9', 'Cython==0.29.28', 'dcor==0.5.3', 'dill==0.3.4', 'dipy==1.4.1',
                      'fury==0.2.0', 'matplotlib==2.2.5', 'nibabel==3.0.2', 'nipype==1.8.5', 'numpy==1.18.5',
                      'opencv_python==4.7.0.68', 'openpyxl==3.0.0', 'pandas==1.3.5', 'paramiko==2.10.1',
                      'scikit_image==0.19.3', 'scikit_learn==0.22.2.post1', 'scipy==1.7.3',
                      'setuptools==46.1.3', 'vtk==8.1.2', 'xlrd==2.0.1', 'XlsxWriter==3.0.3'],
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