import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="restaurant-reviews-allergy",
    version="0.0.1",
    author="Kristen Keller",
    description="Parse restaurant reviews.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_dir={'restaurant-reviews-allergy': 'pkg/restaurant-reviews-allergy'}
)
