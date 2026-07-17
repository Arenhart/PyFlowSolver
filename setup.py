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
        "numpy>=1.23.1",
        "numba>=0.56.2",
        "scipy>=1.9",       # sparse setup algebra for the multigrid solver
        "porespy>=2.3.0",
        "pyedt>=0.1.4",
        "pyamg>=5.0",       # AMG benchmark backend for MultigridSolver
    ],
    )
