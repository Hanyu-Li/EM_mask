from setuptools import setup

entry_points = {}

setup(name='em_mask',
      packages=['em_mask'],
      entry_points=entry_points,
      include_package_data=True,
      version='0.0.1',
      install_requires = [
            'ffn'
      ],
)
