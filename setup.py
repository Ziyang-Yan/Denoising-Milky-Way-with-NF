from setuptools import setup, find_packages

setup(
    name="denoising_milky_way",
    version="0.1.0",
    description="A package for denoising the Milky Way using normalizing flows.",
    author="Ziyang Yan",
    author_email="Ziyang.yan.17@ucl.ac.uk",
    url="https://github.com/Ziyang-Yan/Denoising-Milky-Way-with-NF",
    packages=find_packages(where="denoising_milky_way"),
    package_dir={"": "denoising_milky_way"},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "seaborn",
        "pandas",
        "torch",
        "scikit-learn",
        "tqdm",
        "astropy",
        "zuko",
        "astroML",
        "corner"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
