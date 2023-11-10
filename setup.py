import setuptools

setuptools.setup(
    name="zjb-main",
    version="0.2.5",
    install_requires=[
        "zjb-framework @ https://github.com/ZJBGroup/ZJB-Framework/tarball/main",
        "numba",
        "mako",
        "brainpy",
        "jaxlib",
        "matplotlib",
        "scikit-learn",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
