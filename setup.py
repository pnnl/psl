from setuptools import setup

requirements = [
    # package requirements go here
]

setup(
    name='psl',
    description="dynamic Systems Library in Python",
    license="MIT",
    author="Aaron Tuor, Jan Drgona, Elliott Skomski, Soumya Vasisht",
    author_email='aarontuor@gmail.com',
    url='https://github.com/aarontuor/psl',
    packages=['psl'],
    package_data={'psl': ['psl/parameters/buildings/*']},
    entry_points={
        'console_scripts': [
            'psl=psl.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='psl',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], include_package_data=True
)
