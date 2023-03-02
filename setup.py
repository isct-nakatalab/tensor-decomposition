import setuptools


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensordec",
    version="0.0.1",
    author="chimi",
    author_email="abc@example.com",
    description="It's pip... with git.",
    long_description=long_description,
    url="https://github.com/tokyotech-nakatalab/tensor_decomposition",
    install_requires=_requires_from_file("requirements.txt"),
    packages=setuptools.find_packages("."),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
