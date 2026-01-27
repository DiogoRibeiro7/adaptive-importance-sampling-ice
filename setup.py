"""Setup script for Safe-ICE package."""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from pyproject.toml or define it here
version = "0.1.0"

# Core dependencies
install_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "typing-extensions>=4.0.0;python_version<'3.10'",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "pytest-xdist>=2.5.0",
        "pytest-benchmark>=3.4.1",
        "black>=22.0.0",
        "isort>=5.10.0",
        "ruff>=0.0.261",
        "mypy>=0.991",
        "pre-commit>=2.20.0",
    ],
    "viz": [
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autobuild>=2021.3.14",
        "sphinxcontrib-napoleon>=0.7",
        "nbsphinx>=0.8.9",
        "myst-parser>=0.18.0",
    ],
    "perf": [
        "numba>=0.56.0",
        "psutil>=5.9.0",
    ],
    "all": [],  # Will be populated below
}

# 'all' includes everything
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="safe-ice",
    version=version,
    author="Safe-ICE Contributors",
    author_email="",
    description="Safe Cross-Entropy-Based Importance Sampling for Rare Event Simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-importance-sampling-ice",
    project_urls={
        "Documentation": "https://safe-ice.readthedocs.io/",
        "Source": "https://github.com/yourusername/adaptive-importance-sampling-ice",
        "Tracker": "https://github.com/yourusername/adaptive-importance-sampling-ice/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "safe-ice=safe_ice.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "safe_ice": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "importance-sampling",
        "rare-events",
        "reliability-analysis",
        "monte-carlo",
        "cross-entropy",
        "structural-reliability",
        "probability",
        "simulation",
    ],
)