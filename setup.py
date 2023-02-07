from setuptools import setup

setup(
    name="psl",
    version="1.2",
    description="dynamic Systems Library in Python",
    license="MIT",
    author="Aaron Tuor, Jan Drgona, Elliott Skomski, Soumya Vasisht",
    author_email="aarontuor@gmail.com",
    url="https://gitlab.pnnl.gov/dadaist/psl",
    packages=["psl"],
    package_data={"psl": ["psl/parameters/buildings/*"]},
    entry_points={"console_scripts": ["psl=psl.cli:cli"]},
    # install_requires=[
    #     "gym",
    #     "matplotlib",
    #     # "neuromancer" currently imported by a single script causing a circular dependency between psl and neuromancer
    #     "numpy",
    #     "pandas",
    #     "pyts",
    #     "scipy",
    # ],
    keywords="psl",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    
)