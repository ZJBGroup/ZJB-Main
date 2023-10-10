import setuptools

setuptools.setup(
    name="zjb-main",
    version="0.2.0",
    install_requires=[
        "zjb-framework>=0.2.0",
        "numba",
        "mako",
        "brainpy",
        "jaxlib",
        "matplotlib",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
