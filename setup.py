import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="parzen_estimator",
    version="0.5.5",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/parzen_estimator",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    platforms=["Linux"],
    install_requires=requirements,
    include_package_data=True,
)
