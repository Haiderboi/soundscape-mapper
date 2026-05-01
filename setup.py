"""Setup configuration for Soundscape Mapper package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="soundscape-mapper",
    version="1.0.0",
    author="Soundscape Mapper Contributors",
    description=(
        "Ecoacoustic GIS mapping and analysis platform for "
        "environmental sound analysis"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Haiderboi/soundscape-mapper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "soundscape-process=scripts.process_batch:main",
            "soundscape-maps=scripts.generate_maps:main",
            "soundscape-export=scripts.export_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
