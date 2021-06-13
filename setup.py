import setuptools

setuptools.setup(
    name="MarketMomentum",
    version="1.1.0",
    author="Lee Cote",
    author_email="brian.lee.cote@gmail.com",
    description="Outlier detection model that stratifies NxM Array into levels of outlierness",
    url="https://github.com/LeeCote94/Market-Learning/",
    project_urls={
        "Bug Tracker": "https://github.com/LeeCote94/Market-Learning/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    python_requires=">=3.6",
)