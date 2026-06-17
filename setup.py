import setuptools

setuptools.setup(
    name="pyflowsolver",
    version="0.1",
    author="LTrace technologies",
    description="Flow solver",
    packages=["pyflowsolver"],
    url='https://github.com/yourusername/your_project',
    install_requires=[
        "pytest>=7.4.4",
        "numpy==1.26.4",
        "numba==0.60.0",
        "porespy>=2.3.0",
        "pyedt==0.1.5"
    ],
    )
