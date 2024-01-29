import setuptools

setuptools.setup(
    name="zjb-main",
    version="0.2.5",
    description="Zhejiang Lab's digital twin brain platform",
    author="ZJB Group",
    url="https://github.com/ZJBGroup/ZJB-Main",
    license="GPL-3.0",
    install_requires=[
        "zjb-framework @ https://github.com/ZJBGroup/ZJB-Framework/tarball/main",
        "numba",
        "mako",
        "brainpy",
        "jaxlib",
        "matplotlib",
        "scikit-learn",
        "mne",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
