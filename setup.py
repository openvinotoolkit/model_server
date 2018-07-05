from setuptools import setup


setup(
    name='ie_serving',
    version=0.1,
    description="DLDT inference server",
    long_description="""DLDT inference server""",
    keywords='',
    author_email='',
    packages=['ie_serving', 'ie_serving.server', 'ie_serving.models', 'ie_serving.tensorflow_serving_api'],
    include_package_data=True,
    zip_safe=False,
    install_requires=["grpcio", "grpcio-tools", "numpy", "protobuf", "tensorflow"],
    entry_points={
        'console_scripts': [
            'ie_serving = ie_serving.main:main',
        ]
    },
)
