import os
from setuptools import setup, find_packages

setup(
    name='yolo_detectAPI',
    version='5.7.1.1',
    description='Detect API',
    long_description='This is a API for yolov5 version7 detect.py, new-version 5.7.1 allow user set <conf_thres> and '
                     '<iou_thres>',
    license='GPL Licence',
    author='Da Kuang',
    author_email='dakuang2002@126.com',
    py_modeles = '__init__.py',
    packages=find_packages(),
    pakages=['yolo_detectAPI'],
    include_package_data=True,
    readme = 'README.md',
    python_requires='>=3.8',
    url = 'http://blogs.kd-mercury.xyz/',
    install_requires=['matplotlib>=3.2.2', 'numpy>=1.18.5', 'opencv-python>=4.1.1',
                      'Pillow>=7.1.2', 'PyYAML>=5.3.1', 'requests>=2.23.0', 'scipy>=1.4.1',
                      'thop>=0.1.1', 'torch>=1.7.0', 'torchvision>=0.8.1', 'tqdm>=4.64.0',
                      'tensorboard>=2.4.1', 'pandas>=1.1.4', 'seaborn>=0.11.0',
                      'ipython>=8.3.0', 'psutil>=5.9.4'],
    data_files=['export.py'],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Development Status :: 3 - Alpha"
    ],
    scripts=[],
)