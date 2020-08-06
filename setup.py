from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='slip',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="dynamic Systems Library in Python",
    license="MIT",
    author="Aaron Tuor",
    author_email='aarontuor@gmail.com',
    url='https://github.com/aarontuor/slip',
    packages=['slip'],
    package_data={'slip': 'slip/parameters/buildings/*'},
    entry_points={
        'console_scripts': [
            'slip=slip.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='slip',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], include_package_data=True
)
